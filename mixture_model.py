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
    # Define the interval
    x_min = threshold + 0.01 if threshold == 0 else threshold
    x_max = data_extreme.max().item()
    h = (x_max - x_min) / n_intervals

    x_values = torch.linspace(x_min, x_max, n_intervals)

    p_x = (1 / z_scale_extreme) * (1 + z_shape_extreme * (x_values / z_scale_extreme)) ** (-1 / z_shape_extreme - 1)
    q_x = (1 / z_scale_extreme.exp()) * (1 + z_shape_extreme.pow(2) * (x_values / z_scale_extreme.exp())) ** (-1 / z_shape_extreme.pow(2) - 1)

    kl_values = p_x * torch.log(p_x / q_x)

    kl_gpd = h * (kl_values[0] / 2 + kl_values[1:-1].sum() + kl_values[-1] / 2)
    return kl_gpd.mean()

# Function to classify observations into normal and extreme based on threshold
def classify_observations(y, threshold):
    normal_mask = y.abs() <= threshold
    extreme_mask = y.abs() > threshold
    return normal_mask, extreme_mask

# Adjusted Encoder with additional linear layers and dropout
class Encoder(nn.Module):
    def __init__(self, input_dim, lstm_output_dim, latent_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_output_dim, num_layers=10, batch_first=True, dropout=0.2)
        
        # Additional fully connected layers with dropout
        self.fc1 = nn.Linear(lstm_output_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        
        # Gaussian parameters for normal observations
        self.mean_layer_normal = nn.Linear(lstm_output_dim, latent_dim)
        self.logvar_layer_normal = nn.Linear(lstm_output_dim, latent_dim)
        
        # GPD parameters for extreme observations
        self.scale_layer_extreme = nn.Linear(lstm_output_dim, latent_dim)
        self.shape_layer_extreme = nn.Linear(lstm_output_dim, latent_dim)

    def forward(self, packed_x):
        # lstm_out, _ = self.lstm(X)
        packed_lstm_out, _ = self.lstm(packed_x) 
        lstm_out, length_out = torch.nn.utils.rnn.pad_packed_sequence(packed_lstm_out)   
        lstm_out = lstm_out[-1, :, :]  # Take the last time step output
        
        # Pass through additional layers
        hidden = F.relu(self.fc1(lstm_out))
        hidden = self.dropout(F.relu(self.fc2(hidden)))
        
        z_mean_normal = self.mean_layer_normal(hidden)
        z_log_var_normal = self.logvar_layer_normal(hidden)
        z_scale_extreme = torch.exp(self.scale_layer_extreme(hidden))
        z_shape_extreme = self.shape_layer_extreme(hidden)
        return z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme

# Adjusted Decoder with additional linear layers and dropout
class Decoder(nn.Module):
    def __init__(self, latent_dim, future_steps):
        super(Decoder, self).__init__()
        self.dense1 = nn.Linear(latent_dim, 500)
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 500)
        self.dropout = nn.Dropout(0.2)
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
        # self.pi_gaussian = nn.Parameter(torch.tensor(0.9))  # Weight for Gaussian
        # self.pi_gpd = nn.Parameter(torch.tensor(0.1))       # Weight for GPD
        self.pi_params = nn.Parameter(torch.tensor([3.0, 0.01]))

    def reparameterize(self, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme):
        choice = torch.rand(z_mean_normal.size(0))  # Batch size
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

        # Gaussian loss for normal observations
        R_gaussian = F.mse_loss(reconstructed[normal_mask], y[normal_mask], reduction='mean')
        KL_gaussian = -0.5 * torch.sum(1 + z_log_var_normal - z_mean_normal.pow(2) - z_log_var_normal.exp(), dim=-1).mean()
        Loss_gaussian = pi_gaussian * (R_gaussian + self.beta * KL_gaussian)
        
        # GPD loss for extreme observations
        R_gpd = F.mse_loss(reconstructed[extreme_mask], y[extreme_mask], reduction='mean')
        KL_gpd = -0.5 * torch.sum(1 + z_scale_extreme - z_shape_extreme.pow(2) - z_scale_extreme.exp(), dim=-1).mean()
        # KL_gpd = kl_divergence_gpd_trapezoidal(z_scale_extreme, z_shape_extreme, threshold, y[extreme_mask])
        Loss_gpd = pi_gpd * (R_gpd + self.beta * KL_gpd)
        
        # Mixture distribution loss (comparison with full time series)
        R_mixture = F.mse_loss(reconstructed, y, reduction='mean')
        Loss_mixture = R_mixture + Loss_gaussian + Loss_gpd
        return Loss_mixture

# Define RMSE evaluation function
def rmse(predictions, targets):
    return ((predictions - targets) ** 2).mean().sqrt()