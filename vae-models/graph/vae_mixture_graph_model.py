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

# # Custom GPD reparameterization function
# def reparameterize_gpd(scale, shape, size):
#     uniform_sample = torch.rand(size).to(device)
#     return scale / shape * ((1 - uniform_sample) ** (-shape) - 1)

def reparameterize_gpd(mean, scale, shape):
    """    
    Args:
        mean (float or tensor): location parameter
        scale (float or tensor): scale parameter (> 0)
        shape (float or tensor): shape parameter
    
    Returns:
        torch.Tensor: GPD samples
    """
    # mean = torch.tensor(mean, device=device)
    # scale = torch.tensor(scale, device=device)
    # shape = torch.tensor(shape, device=device)

    u = torch.rand_like(mean)
    u = torch.clamp(u, min=1e-6, max=1 - 1e-6)

    z = mean + (scale / shape) * ((1 - u) ** (-shape) - 1)

    return z

# def kl_gpd(z_mean, z_shape, z_scale, xi0=0.1, sigma0=1.0):
#     """
#     KL divergence between GPD(z_shape, z_scale) and fixed GPD(xi0=0.1, sigma0=1).
#     Assumes same location parameter μ.
#     """
#     xi = z_shape.detach().cpu().numpy()
#     sigma = z_scale.detach().cpu().numpy()
#     mean = z_mean.detach().cpu().numpy()

#     print("xi shape: ", xi.shape)

#     # term3: numerical derivative wrt α at α=0

#     z = (sigma*xi0 - sigma0*xi) / (sigma*xi0)

#     beta_val = beta(1/xi0 - alpha, alpha + 1)
#     hyper_val = hyper((1/xi0 - alpha, -alpha), (1/xi0 + 1,), z)

#     expr = (sigma*xi0/(sigma0*mean + 1e-10))**alpha * beta_val * hyper_val

#     # Lấy đạo hàm tại alpha = 0
#     d_expr = diff(expr, alpha)
#     f = lambdify((), d_expr.subs(alpha, 0), modules="mpmath")
#     val = float(f())

#     # Hệ số ngoài
#     coeff = (1/xi0 + 1) * (1/sigma0)

#     # Totsigma0l KL
#     kl = log(sigma0 / sigma) + (1/sigma0 + 1) * xi0**2 / sigma0 + coeff * val
#     return kl

def kl_gpd_mc(z_mean, z_shape, z_scale, xi0=0.1, sigma0=1.0, num_samples=100):
    # z_shape, z_scale: shape (12, 2, 4, 6624)
    q = GeneralizedPareto(concentration=z_shape, scale=z_scale, loc=z_mean)
    p = GeneralizedPareto(concentration=torch.full_like(z_shape, xi0),
                          scale=torch.full_like(z_scale, sigma0), loc=z_mean)
    
    samples = q.rsample((num_samples,))  # (num_samples, 12, 2, 4, 6624)
    # # Kiểm tra sample có nhỏ hơn loc không
    # invalid_mask = samples < q.loc
    # if invalid_mask.any():
    #     print("mẫu nhỏ hơn loc")


    # print("samples: ", samples[0][0][0][0][0])
    # print("z_shape: ", z_shape[0][0][0][0])

    logq = q.log_prob(samples)
    logp = p.log_prob(samples)

    # # Kiểm tra sample có nhỏ hơn loc không
    has_nan = torch.isinf(logp)
    if has_nan.any():
        print("P nannnnnn")
    has_nan = torch.isinf(logq)
    if has_nan.any():
        print("Q nannnnnn")

    # print("logq shape: ", logq[0][0][0][0][0])
    # print("logp shape: ", logp[0][0][0][0][0])
    # print("logq shape: ", logq.shape)
    # print("logp shape: ", logp.shape)

    # diff = logq - logp
    # valid = (~torch.isnan(diff)) & (~torch.isinf(diff))
    # kl = diff[valid].mean() 
    # print("kl: ", kl)
    kl = (logq - logp).mean()
    return kl

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

        # self._log_dir = self._get_log_dir(dcrnn_kwargs)
        # self._writer = SummaryWriter('runs/' + self._log_dir)
        # log_level = self._kwargs.get('log_level', 'INFO')
        # self._logger = get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        # # LSTM Layer
        # self.lstm = nn.LSTM(input_dim, lstm_output_dim, num_layers=3, batch_first=True)
        # self.layer_norm = nn.LayerNorm(lstm_output_dim)

        # # GCN Layer
        # if self.use_gcn:
        #     self.gcn = pyg_nn.GCNConv(lstm_output_dim, gcn_output_dim)
        #     fc_input_dim = gcn_output_dim  # Input dim for FC layers
        # else:
        #     fc_input_dim = lstm_output_dim  # If not using GCN, input dim is LSTM output dim

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
        # print("##########", x.shape)
        # _, (h_n, _) = self.lstm(x)
        # h_n = self.layer_norm(h_n[-1])  # Take the last hidden state

        # if self.use_gcn:
        #     h_n = self.gcn(h_n, edge_index)

        # x = x.transpose(0, 1)
        _, dcrnn_output = self.dcrnn.encoder(x)
        # dcrnn_output = self.dcrnn(x, y, batches_seen)

        # print("DCRNN output shape: ", dcrnn_output.shape)

        # h_n = dcrnn_output[-1]  # Shape: (batch_size, num_nodes * output_dim)
        h_n = dcrnn_output

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

        return z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

class Decoder(nn.Module):
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
        self.out = nn.Linear(500, output_dim*num_nodes)

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

class VAE(nn.Module):
    def __init__(self, adj_mx, latent_dim, beta=0.001, 
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

        # Init modules if flags are on
        if self.use_d_knn:
            self.d_knn = DKNNImputer3D(
            input_dim=dknn_input_dim,
            t=1.0,
            seq_len=model_kwargs['seq_len'],
            available_node=num_available
            ).to(device)  # D-KNN Imputation
        
        if self.use_gpd and self.use_bernoulli:
            self.pi_params = nn.Parameter(torch.tensor([3.0, 0.01, 0.01]))
        elif self.use_gpd or self.use_bernoulli:
            self.pi_params = nn.Parameter(torch.tensor([3.0, 0.01]))
        else:
            self.pi_params = nn.Parameter(torch.tensor([1.0]))

    def reparameterize(self, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero):
        z_gaussian = reparameterize_gaussian(z_mean_normal, z_log_var_normal)

        if not self.use_gpd and not self.use_bernoulli:
            return z_gaussian  # Tráº£ vá» Gaussian náº¿u khÃ´ng dÃ¹ng GPD/Bernoulli

        # print("self.threshold: ", self.threshold)
        threshold_tensor = torch.full(z_scale_extreme.shape, self.threshold)
        # z_gpd = reparameterize_gpd(z_scale_extreme, z_shape_extreme, z_mean_normal.size()) if self.use_gpd else torch.zeros_like(z_mean_normal)
        z_gpd = reparameterize_gpd(threshold_tensor, z_scale_extreme, z_shape_extreme) if self.use_gpd else torch.zeros_like(z_mean_normal)
        z_bernoulli = reparameterize_bernoulli(z_logits_zero) if self.use_bernoulli else torch.zeros_like(z_mean_normal)

        choice = torch.rand(z_mean_normal.size(0)).to(device)
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

    def forward(self, x, y=None, batches_seen=None):
        # if self.use_d_knn and x_missing is not None:
        #     x_imputed = self.d_knn(x_missing)
        # else:
        #     x_imputed = x_full  
            
        # print("##########", x_imputed.shape)
        z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero = self.encoder(x, y, batches_seen)
        z = self.reparameterize(z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero)
        # print("Z shape: ", z.shape)
        # print("Z mean normal shape: ", z_mean_normal.shape)
        # print("Z log var normal shape: ", z_log_var_normal.shape)
        
        # reconstructed = self.decoder(z)
        reconstructed = []
        for t in range(self.horizon):
            # print("z[t] shape: ", z[t].shape)
            decoder_output = self.decoder(z[t])
            reconstructed.append(decoder_output)
        reconstructed = torch.stack(reconstructed, dim=0)

        return reconstructed, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero

    def loss_function(self, reconstructed, y, z_mean_normal, z_log_var_normal, threshold=None,
                      z_scale_extreme=None, z_shape_extreme=None, z_logits_zero=None, x_full=None, x_imputed=None):
        
        # print("Reconstructed shape: ", reconstructed.shape)
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

        threshold_tensor = torch.full(z_scale_extreme.shape, self.threshold)

        # GPD Loss
        Loss_gpd = 0
        if self.use_gpd:
            # print("z_scale_extreme shape: ", z_scale_extreme.shape)
            # Mask shape: [seq_len, batch_size, num_nodes]
            # mask_node = extreme_mask.any(dim=0)  # [batch_size, num_nodes]
            # print("Mask node shape:", mask_node.shape)
            y_extreme = y[extreme_mask]  # [num_extreme]
            # print("y_extreme shape:", y_extreme.shape)
            if y_extreme.numel() != 0:
                # print("y has extreme values!")

                # batch_size = z_scale_extreme.shape[0]
                # latent_dim = self.encoder.mean_layer_normal.out_features // mask_node.shape[1]
                # num_nodes = mask_node.shape[1]

                # z_scale_extreme = z_scale_extreme.view(batch_size, num_nodes, latent_dim).mean(-1)
                # z_shape_extreme = z_shape_extreme.view(batch_size, num_nodes, latent_dim).mean(-1)

                # # Tìm index (time, batch, node)
                # extreme_idx = extreme_mask.nonzero(as_tuple=True)  # tuple of (time, batch, node)
                # # print("extreme_idx shape:", [x.shape for x in extreme_idx])

                # # Map scale/shape theo batch, node
                # scale_extreme = z_scale_extreme[extreme_idx[1], extreme_idx[2]]  # [num_extreme]
                # shape_extreme = z_shape_extreme[extreme_idx[1], extreme_idx[2]]  # [num_extreme]

                # excess = y_extreme - threshold  # [num_extreme]

                # gpd_nll = torch.mean(
                #     torch.log(scale_extreme)
                #     + (1 + 1 / shape_extreme) * torch.log(1 + shape_extreme * excess / scale_extreme)
                # )
                kld_gpd = kl_gpd_mc(threshold_tensor, z_shape_extreme, z_scale_extreme).mean()
                Loss_gpd = pi_gpd * kld_gpd

                # print("Loss_gpd: ", Loss_gpd)

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
        return total_loss