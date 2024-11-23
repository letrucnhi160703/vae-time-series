import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.optim as optim
import pandas as pd
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

# Load and prepare data
file_path = 'hurdat2.txt'
hurricanes_data = parse_hurricane_data(file_path)

# Prepare X (first 20 steps) and Y (last 4 steps)
X_data = [h[:23] for h in hurricanes_data]  # First 20 steps as input
Y_data = [h[23:] for h in hurricanes_data]  # Last 4 steps as output

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

# Define RMSE evaluation function
def rmse(predictions, targets):
    return ((predictions - targets) ** 2).mean().sqrt()

# Baseline 1: Simple LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, lstm_output_dim, future_steps):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_output_dim, batch_first=True)
        self.fc = nn.Linear(lstm_output_dim, future_steps)

    def forward(self, x):
        # _, (h_n, _) = self.lstm(x)
        # return self.fc(h_n[-1])
        output, (h_n, _) = self.lstm(x)  # output: (batch_size, seq_len, hidden_dim)
        return self.fc(output[:, -1, :])

# Baseline 2: Simple RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, rnn_output_dim, future_steps):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, rnn_output_dim, batch_first=True)
        self.fc = nn.Linear(rnn_output_dim, future_steps)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1])

# Baseline 3: Linear Autoregressive model
class ARModel(nn.Module):
    def __init__(self, input_steps, future_steps):
        super(ARModel, self).__init__()
        self.linear = nn.Linear(input_steps, future_steps)

    def forward(self, x):
        x = x[:, -input_steps:].squeeze(-1)
        return self.linear(x)

# Baseline 4: Transformer
class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, future_steps):
        super(TransformerTimeSeriesModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)  # Embedding layer to map input_dim to d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, future_steps)  # Fully connected layer for final prediction

    def forward(self, x):
        x = self.embedding(x)  # Embed input to d_model
        transformer_output = self.transformer_encoder(x.permute(1, 0, 2))  # Transformer expects (seq_len, batch, d_model)
        output = self.fc(transformer_output[-1])  # Use the last output for future steps prediction
        return output

def train_baseline(model, optimizer, train_loader, epochs):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        total_samples = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = F.mse_loss(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)  # Scale by batch size
            total_samples += X_batch.size(0)  # Track total number of samples
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / total_samples:.4f}")

# Transformer model configuration
d_model = 64
nhead = 8
num_layers = 2

# Initialize baselines
# input_steps = X_train.size(1)
# simple_lstm = SimpleLSTM(input_dim, output_dim, future_steps)
# simple_rnn = SimpleRNN(input_dim, output_dim, future_steps)
# ar_model = ARModel(input_steps, future_steps)
transformer_model = TransformerTimeSeriesModel(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, future_steps=future_steps)

# Optimizers
# optimizer_lstm = optim.Adam(simple_lstm.parameters(), lr=learning_rate)
# optimizer_rnn = optim.Adam(simple_rnn.parameters(), lr=learning_rate)
# optimizer_ar = optim.Adam(ar_model.parameters(), lr=learning_rate)
optimizer_transformer = optim.Adam(transformer_model.parameters(), lr=learning_rate)

# Train baselines
# print("Training Simple LSTM:")
# train_baseline(simple_lstm, optimizer_lstm, train_loader, epochs)

# print("Training Simple RNN:")
# train_baseline(simple_rnn, optimizer_rnn, train_loader, epochs)

# print("Training AR Model:")
# train_baseline(ar_model, optimizer_ar, train_loader, epochs)

print("Training Transformer:")
train_baseline(transformer_model, optimizer_transformer, train_loader, epochs)

# Evaluate with batches
all_truths = []
all_predictions = []

for X_batch, y_batch in test_loader:
    predictions = transformer_model(X_batch)
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


# print(f"Simple LSTM Test RMSE: {average_rmse:.4f}")
# print(f"Simple RNN Test RMSE: {average_rmse:.4f}")
print(f"AR Model Test RMSE: {average_rmse:.4f}")
# print(f"Transformer Test RMSE: {average_rmse:.4f}")

