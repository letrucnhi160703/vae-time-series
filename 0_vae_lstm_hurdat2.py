import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
import numpy as np

# Encoder class
class Encoder(nn.Module):
    def __init__(self, input_dim, lstm_output_dim, latent_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_output_dim, batch_first=True)
        self.mean_layer = nn.Linear(lstm_output_dim, latent_dim)
        self.logvar_layer = nn.Linear(lstm_output_dim, latent_dim)

    def forward(self, packed_x):
        packed_lstm_out, _ = self.lstm(packed_x)
        lstm_out, length_out = torch.nn.utils.rnn.pad_packed_sequence(packed_lstm_out)
        lstm_out = lstm_out[-1, :, :]  # Take the last time step's output
        z_mean = self.mean_layer(lstm_out)
        z_log_var = self.logvar_layer(lstm_out)
        return z_mean, z_log_var

# Decoder class
class Decoder(nn.Module):
    def __init__(self, latent_dim, future_steps):
        super(Decoder, self).__init__()
        self.dense1 = nn.Linear(latent_dim, 500)
        self.out = nn.Linear(500, future_steps)

    def forward(self, z):
        z = F.relu(self.dense1(z))
        return self.out(z)

# VAE class
class VAE(nn.Module):
    def __init__(self, input_dim, lstm_output_dim, latent_dim, future_steps):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, lstm_output_dim, latent_dim)
        self.decoder = Decoder(latent_dim, future_steps)

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
        return forecasting + kl_loss

# Function to parse the data
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

# Load and prepare data
file_path = 'hurdat2.txt'  # Adjust path as needed
hurricanes_data = parse_hurricane_data(file_path)

# Training configuration
input_dim = 1
output_dim = 64
latent_dim = 32
future_steps = 10
epochs = 100
learning_rate = 5e-3

# Train-test split with 80% for training and 20% for testing
train_ratio = 0.8
X_data = []
y_data = []


for wind_speeds in hurricanes_data:
    split_idx = int(len(wind_speeds) * train_ratio)
    X_data.append(torch.tensor(wind_speeds[:split_idx]).float().unsqueeze(1))  # 80% for training
    y_data.append(torch.tensor(wind_speeds[split_idx:split_idx + future_steps]).float())  # 20% for testing

X_length = torch.tensor([len(t) for t in X_data])
y_length = torch.tensor([len(t) for t in y_data])
X_padded = pad_sequence(X_data, batch_first=True)  # Padded training sequences
y_padded = pad_sequence(y_data, batch_first=True)  # Padded test sequences

X_train, X_test, X_length_train, X_length_test, y_train, y_test, y_length_train, y_length_test = train_test_split(X_padded, X_length, y_padded, y_length, test_size=0.2, random_state=42)

X_train_packed = torch.nn.utils.rnn.pack_padded_sequence(X_train, X_length_train, batch_first=True, enforce_sorted=False)
X_test_packed = torch.nn.utils.rnn.pack_padded_sequence(X_test, X_length_test, batch_first=True, enforce_sorted=False)
# y_train_packed = torch.nn.utils.rnn.pack_padded_sequence(y_train, y_length_train, batch_first=True, enforce_sorted=False)
# y_test_packed = torch.nn.utils.rnn.pack_padded_sequence(y_test, y_length_test, batch_first=True, enforce_sorted=False)

# scaler=StandardScaler()
# X_train_data=scaler.fit_transform(np.array(X_train_data).reshape(-1,1))

# Instantiate the model and optimizer
vae = VAE(input_dim=input_dim, lstm_output_dim=output_dim, latent_dim=latent_dim, future_steps=future_steps)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

# Simulate target data for loss calculation on the training set
# y_train = torch.rand((X_padded.size(0), future_steps))

# Training loop
for epoch in range(epochs):
    vae.train()
    optimizer.zero_grad()
    forecasting, z_mean, z_log_var = vae(X_train_packed)
    loss = vae.loss_function(forecasting, y_train, z_mean, z_log_var)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Define RMSE evaluation function
def rmse(predictions, targets):
    return ((predictions - targets) ** 2).mean().sqrt()

# Evaluation on the test set
with torch.no_grad():
    vae.eval()
    test_predictions, _, _ = vae(X_test_packed)
    rmse_score = rmse(test_predictions, y_test)
    print(f"Test RMSE: {rmse_score.item():.4f}")
