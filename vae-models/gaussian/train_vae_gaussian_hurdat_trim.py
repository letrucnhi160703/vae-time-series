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
from vae import VAE
import csv

# Dataset Parsing
def parse_hurricane_data(file_path):
    hurricanes = []
    hurricane_data = []
    header = None

    with open(file_path, 'r') as file:
        for line in file:
            if 'AL' in line:  # Header line for a new hurricane
                # Process the previous hurricane data
                if header and len(hurricane_data) >= 24:
                    wind_speeds = [float(row[6]) for row in hurricane_data if row[6] != '-999']
                    if len(wind_speeds) >= 24:
                        hurricanes.append(wind_speeds[:24])  # Take first 24 steps only
                header = line
                hurricane_data = []
            else:
                hurricane_data.append(line.strip().split(','))

        # Process the last hurricane in the file
        if header and len(hurricane_data) >= 24:
            wind_speeds = [float(row[6]) for row in hurricane_data if row[6] != '-999']
            if len(wind_speeds) >= 24:
                hurricanes.append(wind_speeds[:24])  # Take first 24 steps only

    return hurricanes

# Define RMSE evaluation function
def rmse(predictions, targets):
    return ((predictions - targets) ** 2).mean().sqrt()

# Load and prepare data
file_path = 'hurdat2.txt'
hurricanes_data = parse_hurricane_data(file_path)

# Prepare X (first 20 steps) and Y (last 4 steps)
X_data = [h[:23] for h in hurricanes_data]  # First 23 steps as input
Y_data = [h[23:] for h in hurricanes_data]  # Last 1 steps as output

# # Example output for checking
# for i, (x, y) in enumerate(zip(X_data, Y_data)):
#     print(f"Hurricane {i+1}:")
#     print(f"  Input (X): {x}")
#     print(f"  Output (Y): {y}")
#     print("-" * 50)

# Calculate lengths
X_lengths = [len(x) for x in X_data]  # Length of each input sequence
Y_lengths = [len(y) for y in Y_data]  # Length of each output sequence

# Training configuration
input_dim = 1
output_dim = 64
latent_dim = 32
future_steps = 1
epochs = 100
learning_rate = 5e-3
batch_size = 32

X_train, X_test, X_length_train, X_length_test, y_train, y_test, y_length_train, y_length_test = train_test_split(X_data, X_lengths, Y_data, Y_lengths, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Standardizing data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train.view(-1, 23).numpy()).reshape(-1, 23, 1)  # Fit and transform train
y_train = scaler_y.fit_transform(y_train.numpy().reshape(-1, 1))  # Fit and transform train

X_test = scaler_X.transform(X_test.view(-1, 23).numpy()).reshape(-1, 23, 1)  # Only transform test
y_test = scaler_y.transform(y_test.numpy().reshape(-1, 1))  # Only transform test


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
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
        forecasting, z_mean, z_log_var = vae(X_batch)
        loss = vae.loss_function(forecasting, y_batch, z_mean, z_log_var)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)  # Scale by batch size
        total_samples += X_batch.size(0)  # Track total number of samples
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / total_samples:.4f}")


for X_batch, y_batch in test_loader:
    predictions, _, _ = vae(X_batch)
    all_truths.append(y_batch)
    all_predictions.append(predictions)

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

print(f"Simple LSTM Test RMSE: {average_rmse:.4f}")
