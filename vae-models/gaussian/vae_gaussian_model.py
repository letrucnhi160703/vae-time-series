import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

# Encoder class
class Encoder(nn.Module):
    def __init__(self, input_dim, lstm_output_dim, latent_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_output_dim, batch_first=True)
        self.mean_layer = nn.Linear(lstm_output_dim, latent_dim)
        self.logvar_layer = nn.Linear(lstm_output_dim, latent_dim)

    def forward(self, packed_x):
        # # packed_lstm_out, _ = self.lstm(packed_x)
        # lstm_out, _ = self.lstm(packed_x)
        # # lstm_out, length_out = torch.nn.utils.rnn.pad_packed_sequence(packed_lstm_out)
        # lstm_out = lstm_out[-1, :, :]  # Take the last time step's output
        # z_mean = self.mean_layer(lstm_out)
        # z_log_var = self.logvar_layer(lstm_out)

        _, (h_n, _) = self.lstm(packed_x)  # Use the last hidden state
        h_n = h_n[-1]  # Take the last layecr's hidden state
        z_mean = self.mean_layer(h_n)
        z_log_var = self.logvar_layer(h_n)
        return z_mean, z_log_var

# Decoder class
class Decoder(nn.Module):
    def __init__(self, latent_dim, future_steps):
        super(Decoder, self).__init__()
        self.dense1 = nn.Linear(latent_dim, 500)
        # self.dense2 = nn.Linear(128, 500)
        self.out = nn.Linear(500, future_steps)

    def forward(self, z):
        z = F.relu(self.dense1(z))
        # z = F.relu(self.dense2(z))
        return self.out(z)

# VAE class
class VAE(nn.Module):
    def __init__(self, input_dim, lstm_output_dim, latent_dim, future_steps, beta=0.001):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, lstm_output_dim, latent_dim)
        self.decoder = Decoder(latent_dim, future_steps)
        self.beta = beta

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, X_lstm_input):
        z_mean, z_log_var = self.encoder(X_lstm_input)
        z = self.reparameterize(z_mean, z_log_var)
        forecasting = self.decoder(z)
        return forecasting, z_mean, z_log_var

    def loss_function(self, forecasting, y, z_mean, z_log_var):
        forecasting = F.mse_loss(forecasting, y, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=-1).mean()
        # return forecasting + kl_loss
        return forecasting + self.beta * kl_loss
