import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import dense_to_sparse
import numpy as np
from dcrnn_model import DCRNNModel
from utils import get_logger
from torch.utils.tensorboard import SummaryWriter
import os
import time
from dknn.dknn_model import DKNNImputer3D
from sympy import symbols, diff, log, beta, hyper, simplify
from sympy.abc import alpha
from sympy import lambdify
from dknn.gpd import GeneralizedPareto

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gaussian reparameterization function
def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std

def reparameterize_gpd(mean, scale, shape):
    gpd = GeneralizedPareto(concentration=shape, scale=scale, loc=mean)
    sample = gpd.sample()
    if torch.isnan(sample).any():
        print("sample has nan!!!!")
    return sample

def kl_gpd_mc(z_mean, z_shape, z_scale, xi0=0.1, sigma0=1.0, num_samples=100):
    q = GeneralizedPareto(concentration=z_shape, scale=z_scale, loc=z_mean)
    p = GeneralizedPareto(concentration=torch.full_like(z_shape, xi0),
                          scale=torch.full_like(z_scale, sigma0), loc=z_mean)
    
    samples = q.rsample((num_samples,))

    logq = q.log_prob(samples)
    logp = p.log_prob(samples)

    # num_nan_logq = torch.isnan(logq).sum().item()
    # if num_nan_logq > 0:
    #     print(f"So phan tu NaN trong logq: {num_nan_logq}")
    # num_nan_logp = torch.isnan(logp).sum().item()
    # if num_nan_logp > 0:
    #     print(f"So phan tu NaN trong logp: {num_nan_logp}")
    
    # num_inf_logq = torch.isinf(logq).sum().item()
    # if num_inf_logq > 0:
    #     print(f"So phan tu inf trong logq: {num_inf_logq}")
    # num_inf_logp = torch.isinf(logp).sum().item()
    # if num_inf_logp > 0:
    #     print(f"So phan tu inf trong logp: {num_inf_logp}")

    # valid_mask = ~torch.isinf(logq) & ~torch.isinf(logp) & ~torch.isnan(logq) & ~torch.isnan(logp)
    # logq = logq[valid_mask]
    # logp = logp[valid_mask]

    # if logq.numel() == 0:
    #     print("Khong con phan tu nao hop le sau khi mask Q.")
    # if logp.numel() == 0:
    #     print("Khong con phan tu nao hop le sau khi mask P.")

    kl = (logq - logp).mean()

    if torch.isnan(kl):
        print("KL_gpd: ", kl)

    return kl

def nll_gpd(y, xi, sigma, eps=1e-6):
    mask = ~torch.isnan(y)
    y = y[mask]
    sigma = torch.clamp(sigma, min=eps)
    xi = torch.clamp(xi, min=-1 + eps, max=5.0)
    term = torch.clamp(1 + xi * y / sigma, min=eps)
    log_likelihood = -torch.log(sigma) - (1 + 1/xi) * torch.log(term)
    nll = -log_likelihood
    return torch.mean(nll)

# Bernoulli sampling function
def reparameterize_bernoulli(logits):
    probs = torch.sigmoid(logits)
    # probs = torch.sigmoid(torch.clamp(logits, min=-10, max=10))
    return torch.bernoulli(probs)

# Classify observations
def classify_observations(y, threshold, use_gpd=True, use_bernoulli=True):
    # threshold = threshold.item()
    # print("Threshold: ", threshold)
    # print("y: ", y)
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

class LSTM_GCN_Encoder(nn.Module):
    def __init__(self, adj_mx, latent_dim, use_gpd=True, use_bernoulli=True, **dcrnn_kwargs):
        super(LSTM_GCN_Encoder, self).__init__()

        # Module on/off flags
        # self.use_gcn = use_gcn
        self.use_gpd = use_gpd
        self.use_bernoulli = use_bernoulli
        self._kwargs = dcrnn_kwargs
        self._model_kwargs = dcrnn_kwargs.get('model')

        # DCRNN Layer
        self.dcrnn = DCRNNModel(adj_mx, **self._model_kwargs)
        dcrnn_output_dim = self._model_kwargs['num_nodes'] * self._model_kwargs['rnn_units']
        # print("DCRNN output dim: ", dcrnn_output_dim)
        num_nodes = self._model_kwargs['num_nodes']

        fc_input_dim = dcrnn_output_dim
        # fc_input_dim = 828

        # FC Layers     
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)

        # Gaussian parameters
        self.mean_layer_normal = nn.Linear(64, latent_dim*num_nodes)
        self.logvar_layer_normal = nn.Linear(64, latent_dim*num_nodes)

        # GPD parameters
        if self.use_gpd:
            self.scale_layer_extreme = nn.Linear(64, latent_dim*num_nodes)
            self.shape_layer_extreme = nn.Linear(64, latent_dim*num_nodes)

        # Bernoulli parameter
        if self.use_bernoulli:
            self.logits_layer_zero = nn.Linear(64, latent_dim*num_nodes)

    def forward(self, x, y=None, batches_seen=None):
        last_state, dcrnn_output = self.dcrnn.encoder(x)
        # dcrnn_output = self.dcrnn(x, y, batches_seen)

        # print("DCRNN output shape: ", dcrnn_output.shape)

        # h_n = dcrnn_output[-1]  # Shape: (batch_size, num_nodes * output_dim)
        h_n = dcrnn_output # 12, 2, 4, 828

        # print("h_n shape: ", h_n.shape)

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

        #if torch.isnan(z_scale_extreme).any():
            #print("z_scale_extreme has nan!!!!")

        return z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero, h_n

class Decoder3(nn.Module):
    def __init__(self, latent_dim, output_dim, num_nodes, num_layers):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers

        input_size = latent_dim * num_nodes * num_layers

        # self.fc1 = nn.Linear(latent_dim*num_nodes, 128)
        # self.fc1 = nn.Linear(6624, 128)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 500)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(500, output_dim*num_nodes, bias=True)

    def forward(self, z):
        # print("At Decoder Z shape: ", z.shape)
        z = z.permute(1, 0, 2).reshape(z.size(1), -1)

        z = F.relu(self.fc1(z))
        z = self.dropout(F.relu(self.fc2(z)))
        output = self.out(z)
        # print("Output shape before reshape: ", output.shape)
        # print("Future steps: ", self.future_steps)
        # print("Num nodes: ", self.num_nodes)
        # return output.view(-1, self.future_steps, self.num_nodes).permute(1, 0, 2)
        return output

class Decoder2(nn.Module):
    def __init__(self, latent_dim, output_dim, num_nodes, num_layers):
        super(Decoder, self).__init__()
        input_size = latent_dim * num_nodes * num_layers

        self.fc1 = nn.Linear(input_size, 128)
        # self.norm1 = nn.LayerNorm(128)
        # self.act1 = nn.LeakyReLU(negative_slope=0.01)

        self.fc2 = nn.Linear(128, 500)
        # self.norm2 = nn.LayerNorm(500)
        # self.act2 = nn.ReLU()

        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(500, output_dim * num_nodes, bias=True)

    def forward(self, z):
        z = z.permute(1, 0, 2).reshape(z.size(1), -1)
        # if torch.isnan(z).any() or torch.isinf(z).any():
        #     print("z has invalid values before fc1!")
        #     with open("tensor_debug.txt", "w") as f:
        #         f.write(str(z.tolist()))
        x = self.fc1(z)
        # x = self.norm1(x)
        # x = self.act1(x)

        b = self.fc2(x)
        # c = self.act2(b)
        # c = self.norm2(b)

        d = self.dropout(b)
        output = self.out(d)

        # if torch.isnan(output).any():
        #     print("output has nan!!!!")
        #     print("z before fc1: ", z.mean().item(), z.std().item(), z.max().item(), z.min().item())

        return output

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, num_nodes, num_layers):
        super(Decoder, self).__init__()
        input_size = latent_dim * num_nodes * num_layers

        # ch? 1 l?p linear duy nh?t
        self.out = nn.Linear(input_size, output_dim * num_nodes, bias=True)

    def forward(self, z):
        # reshape l?i z: (batch_size, latent_dim * num_nodes * num_layers)
        # print('z shape', z.shape)
        z = z.permute(1, 0, 2).reshape(z.size(1), -1)
        output = self.out(z)
        return output

class VAE(nn.Module):
    def __init__(self, adj_mx, latent_dim, beta=1.0, 
                 use_d_knn=True, use_gpd=True, use_bernoulli=True, dknn_input_dim=1, num_available=0, threshold=None, **dcrnn_kwargs):
        super(VAE, self).__init__()
        
        # Module on/off flags
        self.use_d_knn = use_d_knn
        # self.use_gcn = use_gcn
        self.use_gpd = use_gpd
        self.use_bernoulli = use_bernoulli

        self.encoder = LSTM_GCN_Encoder(adj_mx, latent_dim, use_gpd, use_bernoulli, **dcrnn_kwargs)

        model_kwargs = dcrnn_kwargs.get('model')
        num_nodes = model_kwargs['num_nodes']
        output_dim = model_kwargs['output_dim']
        num_rnn_layers = model_kwargs['num_rnn_layers']
        self.horizon = model_kwargs['horizon']
        self.decoder = Decoder(latent_dim, output_dim, num_nodes, num_rnn_layers)
        self.beta = beta

        self.threshold = threshold

        self.decoder_xi = nn.Linear(latent_dim * num_nodes * num_rnn_layers, output_dim * num_nodes, bias=True)
        self.decoder_sigma = nn.Linear(latent_dim * num_nodes * num_rnn_layers, output_dim * num_nodes, bias=True)

        # Init modules if flags are on
        if self.use_d_knn:
            self.d_knn = DKNNImputer3D(
            input_dim=dknn_input_dim,
            t=244.0,
            seq_len=model_kwargs['seq_len'],
            available_node=num_available
            ).to(device)  # D-KNN Imputation
        
        if self.use_gpd and self.use_bernoulli:
            self.pi_params = nn.Parameter(torch.tensor([3.0, 0.01, 0.01]))
        elif self.use_gpd or self.use_bernoulli:
            self.pi_params = nn.Parameter(torch.tensor([20.0, 3.0]))
        else:
            self.pi_params = nn.Parameter(torch.tensor([1.0]))

    def reparameterize(self, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero):
        z_gaussian = reparameterize_gaussian(z_mean_normal, z_log_var_normal)

        if not self.use_gpd and not self.use_bernoulli:
            return z_gaussian  # Tráº£ vá» Gaussian náº¿u khÃ´ng dÃ¹ng GPD/Bernoulli

        # print("self.threshold: ", self.threshold)
        threshold_tensor = torch.full(z_scale_extreme.shape, self.threshold).to(device)
        # z_gpd = reparameterize_gpd(z_scale_extreme, z_shape_extreme, z_mean_normal.size()) if self.use_gpd else torch.zeros_like(z_mean_normal)
        z_gpd = reparameterize_gpd(threshold_tensor, z_scale_extreme, z_shape_extreme) if self.use_gpd else torch.zeros_like(z_mean_normal)
        z_gpd = torch.log1p(z_gpd)
        z_bernoulli = reparameterize_bernoulli(z_logits_zero) if self.use_bernoulli else torch.zeros_like(z_mean_normal)

        choice = torch.rand_like(z_mean_normal).to(device)

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

        if torch.isnan(z).any():
            print("z has nan!!!!")
        
        return z

    def forward(self, x, y=None, batches_seen=None):
        z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero, _ = self.encoder(x, y, batches_seen)
        z = self.reparameterize(z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero)
        reconstructed = []
        for t in range(self.horizon):
            # print("z[t] shape: ", z[t].shape)
            decoder_output = self.decoder(z[t])
            reconstructed.append(decoder_output)
        reconstructed = torch.stack(reconstructed, dim=0)

        # reconstructed = self.encoder.dcrnn.decoder(z, labels=y, batches_seen=batches_seen)

        if torch.isnan(reconstructed).any():
            print("reconstructed has nan!!!!")

        # l?y t?ng timestep t nhu decoder chính
        xi_preds, sigma_preds = [], []

        if self.use_gpd:
            for t in range(self.horizon):
                # m?i z_shape_extreme[t]: (batch, num_layers, latent_dim*num_nodes)
                z_shape_t = z_shape_extreme[t].permute(1, 0, 2).reshape(z_shape_extreme[t].size(1), -1)  # (batch, 4*6624)
                z_scale_t = z_scale_extreme[t].permute(1, 0, 2).reshape(z_scale_extreme[t].size(1), -1)

                xi_t = 0.5 * torch.tanh(self.decoder_xi(z_shape_t))
                sigma_t = F.softplus(self.decoder_sigma(z_scale_t)) + 1e-3

                xi_preds.append(xi_t)
                sigma_preds.append(sigma_t)

            xi_extreme_pred = torch.stack(xi_preds, dim=0)        # (12, batch, num_nodes)
            sigma_extreme_pred = torch.stack(sigma_preds, dim=0)  # (12, batch, num_nodes)

            return reconstructed, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero, xi_extreme_pred, sigma_extreme_pred
        return reconstructed, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero, None, None

    def loss_function(self, reconstructed, y, z_mean_normal, z_log_var_normal, threshold=None,
                      z_scale_extreme=None, z_shape_extreme=None, z_logits_zero=None, x_full=None, x_imputed=None, mean_0=None, log_var_0=None,
                      xi_extreme_pred=None, sigma_extreme_pred=None):
        
        # print("Reconstructed shape: ", reconstructed.shape)
        # If not using GPD or Bernoulli, return Gaussian loss
        if not self.use_gpd and not self.use_bernoulli and not self.use_d_knn:
            R_gaussian = F.mse_loss(reconstructed, y, reduction='mean')
            # KL_gaussian = -0.5 * torch.sum(1 + z_log_var_normal - z_mean_normal.pow(2) - z_log_var_normal.exp(), dim=-1).mean()

             # Ensure everything is in same shape
            mean_0_tensor = torch.full_like(z_mean_normal, mean_0)
            log_var_0_tensor = torch.full_like(z_log_var_normal, log_var_0)

            # Convert log-variance to variance
            var_q = torch.exp(z_log_var_normal)
            var_p = torch.exp(log_var_0_tensor)

            term1 = log_var_0_tensor - z_log_var_normal
            term2 = (var_q + (z_mean_normal - mean_0_tensor) ** 2) / (2 * var_p)
            # KL_gaussian = (0.5 * (term1 + term2 - 1)).mean()
            KL_gaussian = (term1 + term2 - 0.5).mean()
            # print("KL_gaussian: ", KL_gaussian)
            
            # KL_gaussian = 0
            Loss_gaussian = R_gaussian + self.beta * KL_gaussian

            return Loss_gaussian, None
            # return R_gaussian, None

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
        mean_0_tensor = torch.full_like(z_mean_normal, mean_0)
        log_var_0_tensor = torch.full_like(z_log_var_normal, log_var_0)

        var_q = torch.exp(z_log_var_normal)
        var_p = torch.exp(log_var_0_tensor)

        term1 = log_var_0_tensor - z_log_var_normal
        term2 = (var_q + (z_mean_normal - mean_0_tensor) ** 2) / (2 * var_p)
        KL_gaussian = (term1 + term2 - 0.5).mean()
        
        # Loss_gaussian = pi_gaussian * (R_gaussian + self.beta * KL_gaussian)
        Loss_gaussian = 0.8 * (R_gaussian + self.beta * KL_gaussian)

        threshold_tensor = torch.full(z_scale_extreme.shape, self.threshold).to(device)

        # # GPD Loss
        # Loss_gpd = 0
        # if self.use_gpd:
        #     KL_gpd = 0
        #     y_extreme = y[extreme_mask]
        #     r_extreme = reconstructed[extreme_mask]
        #     if y_extreme.numel() != 0:
        #         KL_gpd = kl_gpd_mc(threshold_tensor, z_shape_extreme, z_scale_extreme).mean()
        #         # Loss_gpd = pi_gpd * kld_gpd
        #         if torch.isnan(KL_gpd):
        #             print("KL_gpd: ", KL_gpd)
        #         R_gpd = F.mse_loss(r_extreme, y_extreme, reduction='mean')
        #         if torch.isnan(R_gpd):
        #             print("R_gpd: ", R_gpd)
        #         Loss_gpd = pi_gpd * (R_gpd + self.beta * KL_gpd)

        if self.use_gpd:
            y_extreme = y[extreme_mask]
            r_extreme = reconstructed[extreme_mask]
            # print(extreme_mask.shape)
            if y_extreme.numel() != 0:
                # print('y has extreme!!!!!!!')
                y_excess = y_extreme - self.threshold
                y_excess = torch.clamp(y_excess, min=0.0)

                # print("reconstructed:", reconstructed.shape)
                # print("extreme_mask:", extreme_mask.shape)
                # print("y_extreme:", y_extreme.shape)
                # print("y_excess:", y_excess.shape)
                # print("xi:", xi_extreme_pred.shape)
                # print("sigma:", sigma_extreme_pred.shape)
                # print("z_scale_extreme:", z_scale_extreme.shape)

                xi = xi_extreme_pred[extreme_mask]
                sigma = sigma_extreme_pred[extreme_mask]

                nll_val = nll_gpd(y_excess, xi, sigma)
                # print('nll_val: ', nll_val)
                # Loss_gpd = pi_gpd * nll_val

                R_gpd = F.mse_loss(r_extreme, y_extreme, reduction='mean')
                if torch.isnan(R_gpd):
                    print("R_gpd: ", R_gpd)

                extreme_ratio = extreme_mask.float().mean().item() + 1e-6
                alpha_extreme = 1.0 / extreme_ratio
                # Loss_gpd = alpha_extreme * nll_val

                # Loss_gpd = pi_gpd * (R_gpd + alpha_extreme * nll_val)
                Loss_gpd = 0.2 * (R_gpd + alpha_extreme * nll_val)
            else:
                Loss_gpd = 0.0

        # Bernoulli Loss
        Loss_bernoulli = 0
        # if self.use_bernoulli:
        #     mask_node = zero_mask.any(dim=0)  # [batch_size, num_nodes]
        #     y_zero = y[zero_mask]  # [num_extreme]
        #     if y_zero.numel() != 0:
        #         batch_size = z_logits_zero.shape[0]
        #         latent_dim = self.encoder.logits_layer_zero.out_features // mask_node.shape[1]
        #         num_nodes = mask_node.shape[1]

        #         z_logits_zero = z_logits_zero.view(batch_size, num_nodes, latent_dim).mean(-1)

        #         zero_idx = zero_mask.nonzero(as_tuple=True)

        #         logits_zero = z_logits_zero[zero_idx[1], zero_idx[2]]
        #         # print('Logits zero:', logits_zero)
        #         bernoulli_nll = -torch.mean(y_zero * torch.log(torch.sigmoid(logits_zero)) + (1 - y_zero) * torch.log(1 - torch.sigmoid(logits_zero)))
        #         # print('Bernoulli NLL:', bernoulli_nll)
        #         Loss_bernoulli = pi_bernoulli * bernoulli_nll

        # D-KNN Loss
        Loss_d_knn = 0
        if self.use_d_knn and x_full is not None and x_imputed is not None:
            Loss_d_knn = 0.1 * F.mse_loss(x_full, x_imputed, reduction='mean')

        # print ("Loss_bernoulli: ", Loss_bernoulli)
        total_loss = Loss_gaussian + Loss_gpd + Loss_bernoulli + Loss_d_knn
        return total_loss, Loss_gpd