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
from mixture_model import VAE
import csv

# Dataset Parsing
def parse_hurricane_data(file_path, min_length=24):
    hurricanes = []
    hurricane_data = []
    header = None

    with open(file_path, 'r') as file:
        for line in file:
            if 'AL' in line:  # Header line for a new hurricane
                if header and len(hurricane_data) >= min_length:
                    wind_speeds = [float(row[6]) for row in hurricane_data if row[6] != '-999']
                    if len(wind_speeds) >= min_length:
                        hurricanes.append(wind_speeds)
                header = line
                hurricane_data = []
            else:
                hurricane_data.append(line.strip().split(','))

        # Process the last hurricane in the file
        if header and len(hurricane_data) >= min_length:
            wind_speeds = [float(row[6]) for row in hurricane_data if row[6] != '-999']
            if len(wind_speeds) >= min_length:
                hurricanes.append(wind_speeds)

    return hurricanes

# Sliding Window Data Preparation
def create_sliding_windows(hurricanes, window_length=4, predict_steps=1):
    X_data, Y_data = [], []
    for hurricane in hurricanes:
        for i in range(len(hurricane) - window_length - predict_steps + 1):
            # Input: window_length steps
            X_data.append(hurricane[i:i+window_length])
            # Output: predict_steps steps
            Y_data.append(hurricane[i+window_length:i+window_length+predict_steps])
    return X_data, Y_data

# Define RMSE evaluation function
def rmse(predictions, targets):
    return ((predictions - targets) ** 2).mean().sqrt()

# Load and prepare data
file_path = 'hurdat2.txt'
hurricanes_data = parse_hurricane_data(file_path, min_length=24)

# Create Sliding Windows
window_length = 4
predict_steps = 1

# Training configuration
input_dim = 1
output_dim = 64
latent_dim = 32
future_steps = 1

epochs = 20
learning_rate = 5e-5
batch_size = 16
threshold = 95

# epochs = 20
# learning_rate = 0.001
# batch_size = 16
# threshold = 95

X_data, Y_data = create_sliding_windows(hurricanes_data, window_length, predict_steps)

# Convert to NumPy arrays
X_data = np.array(X_data, dtype=np.float32)
Y_data = np.array(Y_data, dtype=np.float32)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

# Standardizing Data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train.reshape(-1, window_length)).reshape(-1, window_length, 1)
y_train = scaler_y.fit_transform(y_train)

X_test = scaler_X.transform(X_test.reshape(-1, window_length)).reshape(-1, window_length, 1)
y_test = scaler_y.transform(y_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model and optimizer
vae = VAE(input_dim=input_dim, lstm_output_dim=output_dim, latent_dim=latent_dim, future_steps=future_steps)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)


# Evaluate with batches
all_truths = []
all_predictions = []

for epoch in range(epochs):
    vae.train()
    epoch_loss = 0
    total_samples = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        reconstructed,  z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme = vae(X_batch)
        loss = vae.loss_function(reconstructed, y_batch, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, threshold)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)  # Scale by batch size
        total_samples += X_batch.size(0)  # Track total number of samples
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / total_samples:.4f}")


for X_batch, y_batch in test_loader:
    reconstructed, _, _, _, _ = vae(X_batch)
    all_truths.append(y_batch)
    all_predictions.append(reconstructed)

# Concatenate all predictions and truths
all_truths = torch.cat(all_truths, dim=0)
all_predictions = torch.cat(all_predictions, dim=0)

# Calculate RMSE over the entire dataset
average_rmse = rmse(all_predictions, all_truths).item()

# Write all predictions vs truth to CSV
with open('predictions_vs_truth.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Truth", "Predictions"])
    for truth, prediction in zip(all_truths, all_predictions):
        writer.writerow([truth.item(), prediction.item()])

print(f"Test RMSE: {average_rmse:.4f}")
