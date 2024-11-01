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

# Dataset Parsing
def parse_hurricane_data(file_path):
    hurricanes = []
    hurricane_data = []
    header = None

    with open(file_path, 'r') as file:
        for line in file:
            if 'AL' in line:
                if header and len(hurricane_data) > 24:
                    wind_speeds = [float(row[6]) for row in hurricane_data if row[6] != '-999']
                    if len(wind_speeds) > 24:
                        hurricanes.append(wind_speeds[:100])  # Truncate to 100 if longer
                header = line
                hurricane_data = []
            else:
                hurricane_data.append(line.strip().split(','))

    if header and len(hurricane_data) > 24:
        wind_speeds = [float(row[6]) for row in hurricane_data if row[6] != '-999']
        if len(wind_speeds) > 24:
            hurricanes.append(wind_speeds[:100])

    return hurricanes

# Adjusted Encoder with additional linear layers and dropout
class Encoder(nn.Module):
    def __init__(self, input_dim, lstm_output_dim, latent_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_output_dim, num_layers=10, batch_first=True, dropout=0.1)
        
        # Additional fully connected layers with dropout
        self.fc1 = nn.Linear(lstm_output_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        
        # Gaussian parameters for normal observations
        self.mean_layer_normal = nn.Linear(64, latent_dim)
        self.logvar_layer_normal = nn.Linear(64, latent_dim)
        
        # GPD parameters for extreme observations
        self.scale_layer_extreme = nn.Linear(64, latent_dim)
        self.shape_layer_extreme = nn.Linear(64, latent_dim)

    def forward(self, X):
        lstm_out, _ = self.lstm(X)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step output
        
        # Pass through additional layers
        hidden = F.relu(self.fc1(lstm_out))
        hidden = self.dropout(F.relu(self.fc2(hidden)))
        
        z_mean_normal = self.mean_layer_normal(hidden)
        z_log_var_normal = self.logvar_layer_normal(hidden)
        z_scale_extreme = torch.exp(self.scale_layer_extreme(hidden))  # Ensure scale is positive
        z_shape_extreme = self.shape_layer_extreme(hidden)  # Shape can be positive or negative
        return z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme

# Adjusted Decoder with additional linear layers and dropout
class Decoder(nn.Module):
    def __init__(self, latent_dim, future_steps):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 500)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(500, future_steps)

        # Final FCN layer
        self.final_fcn = nn.Linear(future_steps, future_steps)  # Output same size as future_steps

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = self.dropout(F.relu(self.fc2(z)))
        output = self.out(z)
        final_output = self.final_fcn(output)  # Final FCN to produce the final prediction
        return final_output

# VAE with Mixture Distribution
class VAE(nn.Module):
    def __init__(self, input_dim, lstm_output_dim, latent_dim, future_steps, beta=1.0):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, lstm_output_dim, latent_dim)
        self.decoder = Decoder(latent_dim, future_steps)
        self.beta = beta
        
        # Initialize mixture distribution weights
        self.pi_gaussian = nn.Parameter(torch.tensor(0.9))  # Weight for Gaussian
        self.pi_gpd = nn.Parameter(torch.tensor(0.1))       # Weight for GPD

    def reparameterize(self, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme):
        choice = torch.rand(z_mean_normal.size(0))  # Batch size
        z = torch.empty_like(z_mean_normal)

        z_gaussian = reparameterize_gaussian(z_mean_normal, z_log_var_normal)
        z_gpd = reparameterize_gpd(z_scale_extreme, z_shape_extreme, z_mean_normal.size())

        for i in range(z.size(0)):
            if choice[i] < self.pi_gaussian:
                z[i] = z_gaussian[i]
            else:
                z[i] = z_gpd[i]

        return z

    def forward(self, X):
        z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme = self.encoder(X)
        z = self.reparameterize(z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme

    def loss_function(self, reconstructed, y, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme):
        R_gaussian = F.mse_loss(reconstructed, y, reduction='mean')
        KL_gaussian = -0.5 * torch.sum(1 + z_log_var_normal - z_mean_normal.pow(2) - z_log_var_normal.exp(), dim=-1).mean()
        Loss_gaussian = self.pi_gaussian * (R_gaussian + self.beta * KL_gaussian)
        
        R_gpd = F.mse_loss(reconstructed, y, reduction='mean')
        KL_gpd = -0.5 * torch.sum(1 + z_scale_extreme - z_shape_extreme.pow(2) - z_scale_extreme.exp(), dim=-1).mean()
        Loss_gpd = self.pi_gpd * (R_gpd + self.beta * KL_gpd)
        
        Loss_mixture = Loss_gaussian + Loss_gpd
        return Loss_mixture

# Define RMSE evaluation function
def rmse(predictions, targets):
    return ((predictions - targets) ** 2).mean().sqrt()