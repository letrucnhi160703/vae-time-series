import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import dense_to_sparse
import numpy as np

# Gaussian reparameterization function
def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std

# Custom GPD reparameterization function
def reparameterize_gpd(scale, shape, size):
    uniform_sample = torch.rand(size)
    return scale / shape * ((1 - uniform_sample) ** (-shape) - 1)

# Bernoulli sampling function
def reparameterize_bernoulli(logits):
    probs = torch.sigmoid(logits)
    # probs = torch.sigmoid(torch.clamp(logits, min=-10, max=10))
    return torch.bernoulli(probs)

# Classify observations
def classify_observations(y, threshold, use_gpd=True, use_bernoulli=True):
    normal_mask = (y <= threshold)
    
    if use_gpd:
        extreme_mask = (y > threshold)
    else:
        extreme_mask = None
    
    if use_bernoulli:
        zero_mask = (y == 0)
        normal_mask = normal_mask & (~zero_mask)
        extreme_mask = extreme_mask & (~zero_mask)
    else:
        zero_mask = None

    if use_gpd and use_bernoulli:
        return normal_mask, extreme_mask, zero_mask
    elif use_gpd:
        return normal_mask, extreme_mask
    elif use_bernoulli:
        return normal_mask, zero_mask
    else:
        return normal_mask

# Differentiable KNN
class D_KNN(nn.Module):
    def __init__(self, k=3, tau=1.0):
        super(D_KNN, self).__init__()
        self.k = k
        self.tau = tau

    def soft_knn_weights(self, X, query):
        distances = torch.cdist(query.unsqueeze(0), X)  #Calculate Euclidean distances
        weights = F.softmax(-distances / self.tau, dim=1)  # Softmax to get weights
        return weights

    def forward(self, X_train, y_train, X_missing):
        weights = self.soft_knn_weights(X_train, X_missing)
        topk_values, topk_indices = torch.topk(weights, self.k, dim=1)  # Choose top K weights
        topk_labels = y_train[topk_indices.squeeze()]  # Choose top K labels

        # Impute missing values using top K labels and weights
        imputed_values = torch.sum(topk_labels * topk_values.unsqueeze(-1), dim=1)
        return imputed_values

class LSTM_GCN_Encoder(nn.Module):
    def __init__(self, input_dim, lstm_output_dim, gcn_output_dim, latent_dim, 
                 use_gcn=True, use_gpd=True, use_bernoulli=True):
        super(LSTM_GCN_Encoder, self).__init__()

        # Module on/off flags
        self.use_gcn = use_gcn
        self.use_gpd = use_gpd
        self.use_bernoulli = use_bernoulli

        # self.channel_projection = nn.Linear(input_dim, 45)

        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, lstm_output_dim, num_layers=3, batch_first=True)
        self.layer_norm = nn.LayerNorm(lstm_output_dim)

        # GCN Layer
        if self.use_gcn:
            self.gcn = pyg_nn.GCNConv(lstm_output_dim, gcn_output_dim)
            fc_input_dim = gcn_output_dim  # Input dim for FC layers
        else:
            fc_input_dim = lstm_output_dim  # If not using GCN, input dim is LSTM output dim

        # FC Layers
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)

        # Gaussian parameters
        self.mean_layer_normal = nn.Linear(64, latent_dim)
        self.logvar_layer_normal = nn.Linear(64, latent_dim)

        # GPD parameters
        if self.use_gpd:
            self.scale_layer_extreme = nn.Linear(64, latent_dim)
            self.shape_layer_extreme = nn.Linear(64, latent_dim)

        # Bernoulli parameter
        if self.use_bernoulli:
            self.logits_layer_zero = nn.Linear(64, latent_dim)

    def forward(self, x, edge_index):
        _, (h_n, _) = self.lstm(x)
        h_n = self.layer_norm(h_n[-1])  # Take the last hidden state

        if self.use_gcn:
            h_n = self.gcn(h_n, edge_index)

        # FC Layers
        hidden = F.relu(self.fc1(h_n))
        hidden = self.dropout(F.relu(self.fc2(hidden)))

        z_mean_normal = self.mean_layer_normal(hidden)
        z_log_var_normal = self.logvar_layer_normal(hidden)

        if self.use_gpd:
            z_scale_extreme = torch.exp(self.scale_layer_extreme(hidden))
            z_shape_extreme = torch.clamp(self.shape_layer_extreme(hidden), min=1e-6)
        else:
            z_scale_extreme = None
            z_shape_extreme = None

        if self.use_bernoulli:
            z_logits_zero = self.logits_layer_zero(hidden)
        else:
            z_logits_zero = None

        return z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero

class Decoder(nn.Module):
    def __init__(self, latent_dim, future_steps):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 500)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(500, future_steps)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = self.dropout(F.relu(self.fc2(z)))
        return self.out(z)

class VAE(nn.Module):
    def __init__(self, input_dim, lstm_output_dim, gcn_output_dim, latent_dim, future_steps, beta=0.001, 
                 use_d_knn=True, use_gcn=True, use_gpd=True, use_bernoulli=True):
        super(VAE, self).__init__()
        
        # Module on/off flags
        self.use_d_knn = use_d_knn
        self.use_gcn = use_gcn
        self.use_gpd = use_gpd
        self.use_bernoulli = use_bernoulli

        # Init modules if flags are on
        if self.use_d_knn:
            self.dk_nn = D_KNN(k=3, tau=1.0)  # D-KNN Imputation

        self.encoder = LSTM_GCN_Encoder(input_dim, lstm_output_dim, gcn_output_dim, latent_dim, use_gcn, use_gpd, use_bernoulli)
        self.decoder = Decoder(latent_dim, future_steps)
        self.beta = beta
        
        if self.use_gpd and self.use_bernoulli:
            self.pi_params = nn.Parameter(torch.tensor([3.0, 0.01, 0.01]))
        elif self.use_gpd or self.use_bernoulli:
            self.pi_params = nn.Parameter(torch.tensor([3.0, 0.01]))
        else:
            self.pi_params = nn.Parameter(torch.tensor([1.0]))

    def reparameterize(self, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero):
        z_gaussian = reparameterize_gaussian(z_mean_normal, z_log_var_normal)

        if not self.use_gpd and not self.use_bernoulli:
            return z_gaussian  # Trả về Gaussian nếu không dùng GPD/Bernoulli

        z_gpd = reparameterize_gpd(z_scale_extreme, z_shape_extreme, z_mean_normal.size()) if self.use_gpd else torch.zeros_like(z_mean_normal)
        z_bernoulli = reparameterize_bernoulli(z_logits_zero) if self.use_bernoulli else torch.zeros_like(z_mean_normal)

        choice = torch.rand(z_mean_normal.size(0))
        z = torch.empty_like(z_mean_normal)

        probs = F.softmax(self.pi_params, dim=0)
        # print("Probs: ", probs)
        if self.use_gpd and self.use_bernoulli:
            z[choice < probs[0]] = z_gaussian[choice < probs[0]]
            z[(choice >= probs[0]) & (choice < probs[0] + probs[1])] = z_gpd[(choice >= probs[0]) & (choice < probs[0] + probs[1])]
            z[choice >= probs[0] + probs[1]] = z_bernoulli[choice >= probs[0] + probs[1]]

        elif self.use_gpd:
            z[choice < probs[0]] = z_gaussian[choice < probs[0]]
            z[choice >= probs[0]] = z_gpd[choice >= probs[0]]

        elif self.use_bernoulli:
            z[choice < probs[0]] = z_gaussian[choice < probs[0]]
            z[choice >= probs[0]] = z_bernoulli[choice >= probs[0]]
        
        return z

    def forward(self, x_full, x_missing=None, edge_index=None):
        if self.use_d_knn and x_missing is not None:
            x_imputed = self.dk_nn(x_full, x_full, x_missing)
        else:
            x_imputed = x_full  # Nếu không dùng D-KNN, giữ nguyên x_full

        z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero = self.encoder(x_imputed, edge_index)
        z = self.reparameterize(z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero

    def loss_function(self, reconstructed, y, z_mean_normal, z_log_var_normal, threshold, z_scale_extreme=None, z_shape_extreme=None, z_logits_zero=None, x_full=None, x_missing=None):
        # If not using GPD or Bernoulli, return Gaussian loss
        if not self.use_gpd and not self.use_bernoulli and not self.use_d_knn:
            R_gaussian = F.mse_loss(reconstructed, y, reduction='mean')
            KL_gaussian = -0.5 * torch.sum(1 + z_log_var_normal - z_mean_normal.pow(2) - z_log_var_normal.exp(), dim=-1).mean()
            # KL_gaussian = 0
            Loss_gaussian = R_gaussian + self.beta * KL_gaussian
            return Loss_gaussian

        # Gaussian Loss
        if self.use_gpd and self.use_bernoulli:
            pi_gaussian, pi_gpd, pi_bernoulli = F.softmax(self.pi_params, dim=0)
            normal_mask, extreme_mask, zero_mask = classify_observations(y, threshold, True, True)
        elif self.use_gpd:
            pi_gaussian, pi_gpd = F.softmax(self.pi_params, dim=0)
            normal_mask, extreme_mask = classify_observations(y, threshold, True, False)
        elif self.use_bernoulli:
            pi_gaussian, pi_bernoulli = F.softmax(self.pi_params, dim=0)
            normal_mask, zero_mask = classify_observations(y, threshold, False, True)

        R_gaussian = F.mse_loss(reconstructed[normal_mask], y[normal_mask], reduction='mean')
        KL_gaussian = -0.5 * torch.sum(1 + z_log_var_normal - z_mean_normal.pow(2) - z_log_var_normal.exp(), dim=-1).mean()
        # KL_gaussian = 0
        Loss_gaussian = pi_gaussian * (R_gaussian + self.beta * KL_gaussian)

        # GPD Loss
        Loss_gpd = 0
        if self.use_gpd:
            extreme_mask = extreme_mask.squeeze(-1) # Làm phẳng mask
            y_extreme = y[extreme_mask] # Lọc ra giá trị cực đoan
            if y_extreme.numel() != 0:
                scale_extreme = z_scale_extreme[extreme_mask] # Lọc scale
                shape_extreme = z_shape_extreme[extreme_mask] # Lọc shape
                excess = y_extreme - threshold # Tính y_i - u
                log_scale_extreme = torch.log(scale_extreme)
                p2 = 1 + 1 / shape_extreme
                p3 = torch.log(1 + shape_extreme * excess / scale_extreme)
                gpd_nll = torch.mean(torch.log(scale_extreme) + (1 + 1 / shape_extreme) * torch.log(1 + shape_extreme * excess / scale_extreme))
                # print('Scale extreme:', scale_extreme)
                # print('Shape extreme:', shape_extreme)
                # print('Excess:', excess)
                # print('GPD NLL:', gpd_nll)
                # if gpd_nll < 0:
                #     print('shape_extreme:', shape_extreme)
                Loss_gpd = pi_gpd * gpd_nll

        # Bernoulli Loss
        Loss_bernoulli = 0
        if self.use_bernoulli:
            zero_mask = zero_mask.squeeze(-1)
            y_zero = y[zero_mask]
            if y_zero.numel() != 0:
                logits_zero = z_logits_zero[zero_mask]
                # print('Logits zero:', logits_zero)
                bernoulli_nll = -torch.mean(y_zero * torch.log(torch.sigmoid(logits_zero)) + (1 - y_zero) * torch.log(1 - torch.sigmoid(logits_zero)))
                # print('Bernoulli NLL:', bernoulli_nll)
                Loss_bernoulli = pi_bernoulli * bernoulli_nll

        # D-KNN Loss
        Loss_d_knn = 0
        if self.use_d_knn and x_missing is not None:
            x_imputed = self.dk_nn(x_full, x_full, x_missing)
            Loss_d_knn = 0.1 * F.mse_loss(x_imputed, x_missing, reduction='mean')

        # print ("Loss_bernoulli: ", Loss_bernoulli)
        total_loss = Loss_gaussian + Loss_gpd + Loss_bernoulli + Loss_d_knn
        return total_loss