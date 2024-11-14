import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.optim as optim

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
                        hurricanes.append(wind_speeds[:100])
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
file_path = 'hurdat2.txt'
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
    X_data.append(torch.tensor(wind_speeds[:split_idx]).float().unsqueeze(1))
    y_data.append(torch.tensor(wind_speeds[split_idx:split_idx + future_steps]).float())

X_length = torch.tensor([len(t) for t in X_data])
y_length = torch.tensor([len(t) for t in y_data])
X_padded = pad_sequence(X_data, batch_first=True)  # Padded training sequences
y_padded = pad_sequence(y_data, batch_first=True)  # Padded test sequences

X_train, X_test, X_length_train, X_length_test, y_train, y_test, y_length_train, y_length_test = train_test_split(X_padded, X_length, y_padded, y_length, test_size=0.2, random_state=42)

X_train_packed = torch.nn.utils.rnn.pack_padded_sequence(X_train, X_length_train, batch_first=True, enforce_sorted=False)
X_test_packed = torch.nn.utils.rnn.pack_padded_sequence(X_test, X_length_test, batch_first=True, enforce_sorted=False)

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
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

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
    
# Training function for baselines
def train_baseline(model, optimizer, X_train, y_train, epochs):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = F.mse_loss(predictions, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Transformer model configuration
d_model = 64
nhead = 8
num_layers = 2

# Initialize baselines
input_steps = X_train.size(1)
simple_lstm = SimpleLSTM(input_dim, output_dim, future_steps)
simple_rnn = SimpleRNN(input_dim, output_dim, future_steps)
ar_model = ARModel(input_steps, future_steps)
transformer_model = TransformerTimeSeriesModel(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, future_steps=future_steps)

# Optimizers
optimizer_lstm = optim.Adam(simple_lstm.parameters(), lr=learning_rate)
optimizer_rnn = optim.Adam(simple_rnn.parameters(), lr=learning_rate)
optimizer_ar = optim.Adam(ar_model.parameters(), lr=learning_rate)
optimizer_transformer = optim.Adam(transformer_model.parameters(), lr=learning_rate)

# Train baselines
print("Training Simple LSTM:")
train_baseline(simple_lstm, optimizer_lstm, X_train, y_train, epochs)

print("Training Simple RNN:")
train_baseline(simple_rnn, optimizer_rnn, X_train, y_train, epochs)

print("Training AR Model:")
train_baseline(ar_model, optimizer_ar, X_train, y_train, epochs)

print("Training Transformer:")
train_baseline(transformer_model, optimizer_transformer, X_train, y_train, epochs)

# Evaluation
with torch.no_grad():
    simple_lstm.eval()
    simple_rnn.eval()
    ar_model.eval()
    transformer_model.eval()
    lstm_rmse = rmse(simple_lstm(X_test), y_test)
    rnn_rmse = rmse(simple_rnn(X_test), y_test)
    ar_rmse = rmse(ar_model(X_test), y_test)
    transformer_rmse = rmse(transformer_model(X_test), y_test)

print(f"Simple LSTM Test RMSE: {lstm_rmse.item():.4f}")
print(f"Simple RNN Test RMSE: {rnn_rmse.item():.4f}")
print(f"AR Model Test RMSE: {ar_rmse.item():.4f}")
print(f"Transformer Test RMSE: {transformer_rmse.item():.4f}")
