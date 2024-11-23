import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.optim as optim
import pandas as pd
import csv

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

# Prepare mask tensor
def create_masks(padded_sequences, lengths):
    max_len = padded_sequences.size(1)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

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

# Create masks
train_mask = create_masks(X_train, X_length_train)
test_mask = create_masks(X_test, X_length_test)

# Define RMSE evaluation function
def rmse(predictions, targets):
    return ((predictions - targets) ** 2).mean().sqrt()

# Baseline 1: Simple LSTM model
class MaskedLSTM(nn.Module):
    def __init__(self, input_dim, lstm_output_dim, future_steps):
        super(MaskedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_output_dim, batch_first=True)
        self.fc = nn.Linear(lstm_output_dim, future_steps)

    def forward(self, x, lengths):
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed_x)
        return self.fc(h_n[-1])

# Baseline 2: Simple RNN model
class MaskedRNN(nn.Module):
    def __init__(self, input_dim, rnn_output_dim, future_steps):
        super(MaskedRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, rnn_output_dim, batch_first=True)
        self.fc = nn.Linear(rnn_output_dim, future_steps)

    def forward(self, x, lengths):
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed_x)
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
class MaskedTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, future_steps):
        super(MaskedTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, future_steps)

    def forward(self, x, mask):
        x = self.embedding(x)
        transformer_output = self.transformer_encoder(x.permute(1, 0, 2), src_key_padding_mask=~mask)
        return self.fc(transformer_output[-1])

# Training function
def train_with_mask(model, optimizer, X_train, y_train, lengths, mask, epochs):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train, lengths if hasattr(model, 'lstm') else mask)
        loss = F.mse_loss(predictions, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Transformer model configuration
d_model = 64
nhead = 8
num_layers = 2
input_steps = X_train.size(1)

# Initialize baselines
masked_lstm = MaskedLSTM(input_dim=input_dim, lstm_output_dim=output_dim, future_steps=future_steps)
# masked_rnn = MaskedRNN(input_dim=input_dim, rnn_output_dim=output_dim, future_steps=future_steps)
# ar_model = ARModel(input_steps=input_steps, future_steps=future_steps)
# masked_transformer = MaskedTransformer(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, future_steps=future_steps)

# Optimizers
optimizer_lstm = optim.Adam(masked_lstm.parameters(), lr=learning_rate)
# optimizer_rnn = optim.Adam(masked_rnn.parameters(), lr=learning_rate)
# optimizer_ar = optim.Adam(ar_model.parameters(), lr=learning_rate)
# optimizer_transformer = optim.Adam(masked_transformer.parameters(), lr=learning_rate)

# Train baselines
print("Training Masked LSTM:")
train_with_mask(masked_lstm, optimizer_lstm, X_train, y_train, X_length_train, train_mask, epochs)

# print("Training Masked RNN:")
# train_with_mask(masked_rnn, optimizer_rnn, X_train, y_train, X_length_train, train_mask, epochs)

# print("Training AR Model:")
# train_with_mask(ar_model, optimizer_ar, X_train, y_train, X_length_train, train_mask, epochs)

# print("Training Masked Transformer:")
# train_with_mask(masked_transformer, optimizer_transformer, X_train, y_train, X_length_train, train_mask, epochs)

# Evaluation
def evaluate_with_mask(model, X_test, y_test, lengths, mask):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test, lengths if hasattr(model, 'lstm') else mask)
        test_rmse = rmse(predictions, y_test)
    print(f"Test RMSE: {test_rmse.item():.4f}")
    return predictions

# Save Results to CSV
def save_results_to_csv(y_true, y_pred, file_name="predictions_vs_truth.csv"):
    """
    Save ground truth and predictions to a CSV file.
    """
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Truth", "Predictions"])
        for truth, prediction in zip(y_true, y_pred):
            for true_val, pred_val in zip(truth.tolist(), prediction.tolist()):
                writer.writerow([true_val, pred_val])
    print(f"Results saved to '{file_name}'")

print("Evaluating Masked LSTM:")
predictions = evaluate_with_mask(masked_lstm, X_test, y_test, X_length_test, test_mask)

# print("Evaluating Masked RNN:")
# evaluate_with_mask(masked_rnn, X_test, y_test, X_length_test, test_mask)

# print("Evaluating AR Model:")
# evaluate_with_mask(ar_model, X_test, y_test, X_length_test, test_mask)

# print("Evaluating Masked Transformer:")
# evaluate_with_mask(masked_transformer, X_test, y_test, X_length_test, test_mask)

save_results_to_csv(y_test.cpu().numpy(), predictions.cpu().numpy(), file_name="predictions_vs_truth.csv")
