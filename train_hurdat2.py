import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import mixture_model as model

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

# Load and prepare data
file_path = 'hurdat2.txt'  # Adjust path as needed
hurricanes_data = parse_hurricane_data(file_path)

# Training configuration
input_dim = 1
lstm_output_dim = 128
latent_dim = 16
future_steps = 10
beta = 1.0
epochs = 50
learning_rate = 1e-5
batch_size = 128 
lstm_layers = 10
threshold = 100 

# Train-test split with 80% for training and 20% for testing
train_ratio = 0.8
X_train_data = []
y_test_data = []

# Split the dataset into sequences
for wind_speeds in hurricanes_data:
    split_idx = int(len(wind_speeds) * train_ratio)
    X_train_data.append(torch.tensor(wind_speeds[:split_idx]).float().unsqueeze(1))  # 80% for training
    y_test_data.append(torch.tensor(wind_speeds[split_idx:split_idx + future_steps]).float())  # 20% for testing

X_padded = pad_sequence(X_train_data, batch_first=True)  # Padded training sequences
y_test_padded = pad_sequence(y_test_data, batch_first=True)  # Padded test sequences
y_train = torch.rand((X_padded.size(0), future_steps))  # Simulate target data for loss calculation on the training set

# Instantiate the model and optimizer
vae = model.VAE(input_dim=input_dim, lstm_output_dim=lstm_output_dim, latent_dim=latent_dim, future_steps=future_steps, beta=beta)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

# Data loader for batch processing
train_data = TensorDataset(X_padded, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Training loop with batch loading
for epoch in range(epochs):
    vae.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        reconstructed, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme = vae(X_batch)
        loss = vae.loss_function(reconstructed, y_batch, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

# Evaluation on the test set
with torch.no_grad():
    vae.eval()
    test_predictions, _, _, _, _ = vae(X_padded)
    rmse_score = model.rmse(test_predictions, y_test_padded)
    print(f"Test RMSE: {rmse_score.item():.4f}")
