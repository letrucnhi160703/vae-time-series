import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import mixture_model as model
from sklearn.model_selection import train_test_split

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
lstm_output_dim = 64
latent_dim = 32
future_steps = 10
# beta = 1.0
epochs = 50
learning_rate = 5e-3
# lstm_layers = 10
threshold = 95 

# Train-test split with 80% for training and 20% for testing
train_ratio = 0.8
X_data = []
y_data = []

# Split the dataset into sequences
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

# Instantiate the model and optimizer
vae = model.VAE(input_dim=input_dim, lstm_output_dim=lstm_output_dim, latent_dim=latent_dim, future_steps=future_steps)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

for epoch in range(epochs):
    vae.train()
    optimizer.zero_grad()
    reconstructed, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme = vae(X_train_packed)
    loss = vae.loss_function(reconstructed, y_train, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, threshold)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    pi_gaussian, pi_gpd = F.softmax(vae.pi_params, dim=0)
    print(f"pi_gaussian: {pi_gaussian.item():.4f}, pi_gpd: {pi_gpd.item():.4f}\n")


# Evaluation on the test set
with torch.no_grad():
    vae.eval()
    test_predictions, _, _, _, _ = vae(X_test_packed)
    # test_predictions, _, _, _, _ = vae(X_test)
    rmse_score = model.rmse(test_predictions, y_test)
    print(f"Test RMSE: {rmse_score.item():.4f}")
    test_predictions = test_predictions.numpy()
    if isinstance(y_test, list):
        y_test = torch.tensor(y_test)
    y_test_numpy = y_test.cpu().numpy()

    results_df = pd.DataFrame({
    'Truth': y_test_numpy.flatten(),
    'Predictions': test_predictions.flatten()
    })

    # Xuất ra file CSV
    results_df.to_csv('predictions_vs_truth.csv', index=False)
    print("Results saved to 'predictions_vs_truth.csv'")
