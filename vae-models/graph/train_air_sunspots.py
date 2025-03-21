import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv
import pandas as pd
import torch.optim as optim
from vae_mixture_graph_model import VAE

# Sliding Window Data Preparation
def create_sliding_windows(data, window_length=4, predict_steps=1):
    X_data, Y_data = [], []
    for i in range(len(data) - window_length - predict_steps + 1):
        # Input: window_length steps
        X_data.append(data[i:i+window_length])
        # Output: predict_steps steps
        Y_data.append(data[i+window_length:i+window_length+predict_steps])
    return X_data, Y_data

# Load Dataset
def load_csv_dataset(file_path):
    df = pd.read_csv(file_path)
    if 'value' not in df.columns:
        raise ValueError("The CSV file must contain a 'value' column.")
    values = df['value'].values.astype(np.float32)
    return values

# RMSE Function
def rmse(predictions, targets):
    return torch.sqrt(F.mse_loss(predictions, targets))

# Load your CSV file
# file_path = 'datasets/sunspots.csv'
file_path = '../../datasets/sunspots.csv'
# file_path = '../../datasets/AirPassengers.csv'
# file_path = '../../datasets/LD2011_2014_less.csv'
data = load_csv_dataset(file_path)

# Sliding window parameters
window_length = 4
predict_steps = 1
percentile = 95

# Create Sliding Windows
X_data, Y_data = create_sliding_windows(data, window_length, predict_steps)
threshold = np.percentile(Y_data, percentile)

# Convert to NumPy arrays
X_data = np.array(X_data, dtype=np.float32)
Y_data = np.array(Y_data, dtype=np.float32)

print(f"Số phần tử y bằng 0: {(Y_data == 0).sum().item()}")
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

# Standardizing Data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train.reshape(-1, window_length)).reshape(-1, window_length, 1)
y_train = scaler_y.fit_transform(y_train)

X_test = scaler_X.transform(X_test.reshape(-1, window_length)).reshape(-1, window_length, 1)
y_test = scaler_y.transform(y_test)

threshold_scaled = scaler_y.transform(np.array([[threshold]]))
threshold = threshold_scaled[0, 0]
print(f"Threshold at Q {percentile}% is: {threshold}")

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# DataLoader
batch_size = 16
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training configuration
input_dim = 1
output_dim = 64
latent_dim = 32
epochs = 20
learning_rate = 1e-3

# Instantiate the model and optimizer
vae = VAE(input_dim=input_dim, lstm_output_dim=output_dim, gcn_output_dim=0, latent_dim=latent_dim, future_steps=predict_steps, use_d_knn=False, use_gcn=False, use_gpd=True, use_bernoulli=True)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

# Evaluate with batches
all_truths = []
all_predictions = []

for epoch in range(epochs):
    vae.train()
    epoch_loss = 0
    total_samples = 0
    for X_batch, y_batch in train_loader:
        if (y_batch == 0).sum().item() > 0:
            print(f"Số phần tử y bằng 0: {(y_batch == 0).sum().item()}")
        optimizer.zero_grad()
        forecasting, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero = vae(X_batch, X_batch, None)
        loss = vae.loss_function(forecasting, y_batch, z_mean_normal, z_log_var_normal, threshold, z_scale_extreme, z_shape_extreme, z_logits_zero)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)  # Scale by batch size
        total_samples += X_batch.size(0)  # Track total number of samples
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / total_samples:.4f}")


for X_batch, y_batch in test_loader:
    predictions, _, _, _, _, _ = vae(X_batch, X_batch, None)
    all_truths.append(y_batch)
    all_predictions.append(predictions)

# Concatenate all predictions and truths
all_truths = torch.cat(all_truths, dim=0)
all_predictions = torch.cat(all_predictions, dim=0)

# Calculate RMSE over the entire dataset
average_rmse = rmse(all_predictions, all_truths).item()

# Write all predictions vs truth to CSV
with open('../../visualization/predictions_vs_truth.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Truth", "Predictions"])
    for truth, prediction in zip(all_truths, all_predictions):
        writer.writerow([truth.item(), prediction.item()])

print(f"Test RMSE: {average_rmse:.4f}")