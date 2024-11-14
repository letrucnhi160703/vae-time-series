import torch
import torch.nn as nn
import torch.nn.functional as F

# Gaussian reparameterization function
def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std

# Custom GPD reparameterization function
def reparameterize_gpd(scale, shape, size):
    uniform_sample = torch.rand(size)
    return scale / shape * ((1 - uniform_sample) ** (-shape) - 1)

def kl_divergence_gpd_trapezoidal(z_scale_extreme, z_shape_extreme, threshold, data_extreme, n_intervals=100):
    x_min = threshold + 0.01 if threshold == 0 else threshold
    x_max = data_extreme.max().item()
    h = (x_max - x_min) / n_intervals
    x_values = torch.linspace(x_min, x_max, n_intervals)

    p_x = (1 / z_scale_extreme) * (1 + z_shape_extreme * (x_values / z_scale_extreme)) ** (-1 / z_shape_extreme - 1)
    q_x = (1 / z_scale_extreme.exp()) * (1 + z_shape_extreme.pow(2) * (x_values / z_scale_extreme.exp())) ** (-1 / z_shape_extreme.pow(2) - 1)

    kl_values = p_x * torch.log(p_x / q_x)
    kl_gpd = h * (kl_values[0] / 2 + kl_values[1:-1].sum() + kl_values[-1] / 2)
    return kl_gpd.mean()

def classify_observations(y, threshold):
    normal_mask = y.abs() <= threshold
    extreme_mask = y.abs() > threshold
    return normal_mask, extreme_mask

# Transformer-based Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, nhead, latent_dim):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mean_layer_normal = nn.Linear(d_model, latent_dim)
        self.logvar_layer_normal = nn.Linear(d_model, latent_dim)
        self.scale_layer_extreme = nn.Linear(d_model, latent_dim)
        self.shape_layer_extreme = nn.Linear(d_model, latent_dim)

    def forward(self, x):
        x = self.embedding(x)
        transformer_out = self.transformer_encoder(x.permute(1, 0, 2))  # (seq_len, batch, d_model)
        last_hidden = transformer_out[-1]  # Take the last hidden state
        z_mean_normal = self.mean_layer_normal(last_hidden)
        z_log_var_normal = self.logvar_layer_normal(last_hidden)
        z_scale_extreme = torch.exp(self.scale_layer_extreme(last_hidden))
        z_shape_extreme = self.shape_layer_extreme(last_hidden)
        return z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme

class Decoder(nn.Module):
    def __init__(self, latent_dim, future_steps):
        super(Decoder, self).__init__()
        self.dense1 = nn.Linear(latent_dim, 500)
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 500)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(500, future_steps)
        self.final_fcn = nn.Linear(future_steps, future_steps)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = self.dropout(F.relu(self.fc2(z)))
        output = self.out(z)
        final_output = self.final_fcn(output)
        return final_output

class VAE(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, nhead, latent_dim, future_steps, beta=1.0):
        super(VAE, self).__init__()
        self.encoder = TransformerEncoder(input_dim=input_dim, d_model=d_model, num_layers=num_layers, nhead=nhead, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, future_steps=future_steps)
        self.beta = beta
        self.pi_params = nn.Parameter(torch.tensor([3.0, 0.01]))

    def reparameterize(self, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme):
        choice = torch.rand(z_mean_normal.size(0))
        z = torch.empty_like(z_mean_normal)
        z_gaussian = reparameterize_gaussian(z_mean_normal, z_log_var_normal)
        z_gpd = reparameterize_gpd(z_scale_extreme, z_shape_extreme, z_mean_normal.size())
        for i in range(z.size(0)):
            if choice[i] < F.softmax(self.pi_params, dim=0)[0]:
                z[i] = z_gaussian[i]
            else:
                z[i] = z_gpd[i]
        return z

    def forward(self, X):
        z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme = self.encoder(X)
        z = self.reparameterize(z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme

    def loss_function(self, reconstructed, y, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, threshold):
        pi_gaussian, pi_gpd = F.softmax(self.pi_params, dim=0)
        normal_mask, extreme_mask = classify_observations(y, threshold)
        R_gaussian = F.mse_loss(reconstructed[normal_mask], y[normal_mask], reduction='mean')
        KL_gaussian = -0.5 * torch.sum(1 + z_log_var_normal - z_mean_normal.pow(2) - z_log_var_normal.exp(), dim=-1).mean()
        Loss_gaussian = pi_gaussian * (R_gaussian + self.beta * KL_gaussian)

        R_gpd = F.mse_loss(reconstructed[extreme_mask], y[extreme_mask], reduction='mean')
        KL_gpd = -0.5 * torch.sum(1 + z_scale_extreme - z_shape_extreme.pow(2) - z_scale_extreme.exp(), dim=-1).mean()
        Loss_gpd = pi_gpd * (R_gpd + self.beta * KL_gpd)

        R_mixture = F.mse_loss(reconstructed, y, reduction='mean')
        Loss_mixture = R_mixture + Loss_gaussian + Loss_gpd
        return Loss_mixture

# Define RMSE evaluation function
def rmse(predictions, targets):
    return ((predictions - targets) ** 2).mean().sqrt()