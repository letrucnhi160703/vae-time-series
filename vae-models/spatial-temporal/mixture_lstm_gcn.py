import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import dense_to_sparse

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
    return torch.bernoulli(probs)

# Classify observations
def classify_observations(y, threshold):
    normal_mask = (y.abs() <= threshold)
    extreme_mask = (y.abs() > threshold)
    zero_mask = (y == 0)
    return normal_mask, extreme_mask, zero_mask

class LSTM_GCN_Encoder(nn.Module):
    def __init__(self, input_dim, lstm_output_dim, gcn_output_dim, latent_dim):
        super(LSTM_GCN_Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_output_dim, num_layers=3, batch_first=True)
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        self.gcn = pyg_nn.GCNConv(lstm_output_dim, gcn_output_dim)
        self.fc1 = nn.Linear(gcn_output_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        
        # Gaussian parameters for normal observations
        self.mean_layer_normal = nn.Linear(64, latent_dim)
        self.logvar_layer_normal = nn.Linear(64, latent_dim)
        
        # GPD parameters for extreme observations
        self.scale_layer_extreme = nn.Linear(64, latent_dim)
        self.shape_layer_extreme = nn.Linear(64, latent_dim)
        
        # Bernoulli parameter for zero observations
        self.logits_layer_zero = nn.Linear(64, latent_dim)

    def forward(self, x, edge_index):
        _, (h_n, _) = self.lstm(x)
        h_n = self.layer_norm(h_n[-1])
        h_gcn = self.gcn(h_n, edge_index)
        hidden = F.relu(self.fc1(h_gcn))
        hidden = self.dropout(F.relu(self.fc2(hidden)))

        z_mean_normal = self.mean_layer_normal(hidden)
        z_log_var_normal = self.logvar_layer_normal(hidden)
        z_scale_extreme = torch.exp(self.scale_layer_extreme(hidden))
        z_shape_extreme = torch.clamp(self.shape_layer_extreme(hidden), min=1e-6)
        z_logits_zero = self.logits_layer_zero(hidden)

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
    def __init__(self, input_dim, lstm_output_dim, gcn_output_dim, latent_dim, future_steps, beta=0.001):
        super(VAE, self).__init__()
        self.encoder = LSTM_GCN_Encoder(input_dim, lstm_output_dim, gcn_output_dim, latent_dim)
        self.decoder = Decoder(latent_dim, future_steps)
        self.beta = beta
        self.pi_params = nn.Parameter(torch.tensor([3.0, 0.01, 0.01]))

    def reparameterize(self, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero):
        choice = torch.rand(z_mean_normal.size(0))
        z = torch.empty_like(z_mean_normal)

        z_gaussian = reparameterize_gaussian(z_mean_normal, z_log_var_normal)
        z_gpd = reparameterize_gpd(z_scale_extreme, z_shape_extreme, z_mean_normal.size())
        z_bernoulli = reparameterize_bernoulli(z_logits_zero)

        for i in range(z.size(0)):
            probs = F.softmax(self.pi_params, dim=0)
            if choice[i] < probs[0]:
                z[i] = z_gaussian[i]
            elif choice[i] < probs[0] + probs[1]:
                z[i] = z_gpd[i]
            else:
                z[i] = z_bernoulli[i]

        return z

    def forward(self, x, edge_index):
        z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero = self.encoder(x, edge_index)
        z = self.reparameterize(z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero

    def loss_function(self, reconstructed, y, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero, threshold):
        normal_mask, extreme_mask, zero_mask = classify_observations(y, threshold)

        # Gaussian Loss
        pi_gaussian, pi_gpd, pi_bernoulli = F.softmax(self.pi_params, dim=0)
        R_gaussian = F.mse_loss(reconstructed[normal_mask], y[normal_mask], reduction='mean')
        KL_gaussian = -0.5 * torch.sum(1 + z_log_var_normal - z_mean_normal.pow(2) - z_log_var_normal.exp(), dim=-1).mean()
        Loss_gaussian = pi_gaussian * (R_gaussian + self.beta * KL_gaussian)

        # GPD Loss
        y_extreme = y[extreme_mask]
        scale_extreme = z_scale_extreme[extreme_mask]
        shape_extreme = z_shape_extreme[extreme_mask]
        excess = y_extreme - threshold
        gpd_nll = torch.mean(torch.log(scale_extreme) + (1 + 1 / shape_extreme) * torch.log(1 + shape_extreme * excess / scale_extreme))
        Loss_gpd = pi_gpd * gpd_nll

        # Bernoulli Loss
        logits_zero = z_logits_zero[zero_mask]
        y_zero = y[zero_mask]
        bernoulli_nll = -torch.mean(y_zero * torch.log(torch.sigmoid(logits_zero)) + (1 - y_zero) * torch.log(1 - torch.sigmoid(logits_zero)))
        Loss_bernoulli = pi_bernoulli * bernoulli_nll

        # Total Loss
        total_loss = Loss_gaussian + Loss_gpd + Loss_bernoulli
        return total_loss