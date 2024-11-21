import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Reparameterization for GPD
def reparameterize_gpd(scale, shape, size):
    uniform_sample = torch.rand(size)
    return scale / shape * ((1 - uniform_sample) ** (-shape) - 1)

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, lstm_output_dim, latent_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_output_dim, num_layers=3, batch_first=True)
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        self.fc1 = nn.Linear(lstm_output_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.scale_layer_extreme = nn.Linear(64, latent_dim)
        self.shape_layer_extreme = nn.Linear(64, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = self.layer_norm(h_n[-1])
        hidden = F.relu(self.fc1(h_n))
        hidden = self.dropout(F.relu(self.fc2(hidden)))
        z_scale_extreme = torch.exp(self.scale_layer_extreme(hidden))
        z_shape_extreme = torch.clamp(self.shape_layer_extreme(hidden), min=1e-6)
        return z_scale_extreme, z_shape_extreme

# Decoder
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

# VAE
class VAE(nn.Module):
    def __init__(self, input_dim, lstm_output_dim, latent_dim, future_steps, beta=0.001):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, lstm_output_dim, latent_dim)
        self.decoder = Decoder(latent_dim, future_steps)
        self.beta = beta

    def reparameterize(self, z_scale_extreme, z_shape_extreme):
        z = reparameterize_gpd(z_scale_extreme, z_shape_extreme, z_scale_extreme.size())
        return z

    def forward(self, x):
        z_scale_extreme, z_shape_extreme = self.encoder(x)
        z = self.reparameterize(z_scale_extreme, z_shape_extreme)
        reconstructed = self.decoder(z)
        return reconstructed, z_scale_extreme, z_shape_extreme

    def loss_function(self, reconstructed, y, z_scale_extreme, z_shape_extreme):
        R_gpd = F.mse_loss(reconstructed, y, reduction='mean')
        KL_gpd = torch.mean(torch.log(z_scale_extreme) + 
                            (1 + z_shape_extreme * threshold / z_scale_extreme).log())
        return R_gpd + self.beta * KL_gpd

# Load data
X_train_max, y_train_max, y_val_max, X_test_max, y_test_max = torch.load("data_split.pt", map_location="cpu")

# Configuration
input_dim = 1
output_dim = 64
latent_dim = 32
future_steps = 1
epochs = 20
learning_rate = 0.001
batch_size = 16
threshold = 95

# Instantiate model and optimizer
vae = VAE(input_dim=input_dim, lstm_output_dim=output_dim, latent_dim=latent_dim, future_steps=future_steps)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train_max, y_train_max), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_max, y_test_max), batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(epochs):
    vae.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        reconstructed, z_scale_extreme, z_shape_extreme = vae(X_batch)
        loss = vae.loss_function(reconstructed, y_batch, z_scale_extreme, z_shape_extreme)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
vae.eval()
all_predictions = []
all_truths = []
with torch.no_grad():
    total_rmse = 0
    for X_batch, y_batch in test_loader:
        reconstructed, _, _ = vae(X_batch)
        rmse_score = np.sqrt(mean_squared_error(y_batch.cpu(), reconstructed.cpu()))
        total_rmse += rmse_score
        all_predictions.extend(reconstructed.cpu().numpy())
        all_truths.extend(y_batch.cpu().numpy())
    
    print(f"Test RMSE: {total_rmse / len(test_loader):.4f}")

# Save results
results_df = pd.DataFrame({
    'Truth': np.array(all_truths).flatten(),
    'Predictions': np.array(all_predictions).flatten()
})
results_df.to_csv('predictions_vs_truth_gpd.csv', index=False)
print("Results saved to 'predictions_vs_truth_gpd.csv'")
