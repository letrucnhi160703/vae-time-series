import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
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

# Model Definition
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, lstm_output_dim, future_steps):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_output_dim, batch_first=True)
        self.fc = nn.Linear(lstm_output_dim, future_steps)

    def forward(self, x):
        output, (h_n, _) = self.lstm(x)  # output: (batch_size, seq_len, hidden_dim)
        return self.fc(output[:, -1, :])  # Use the last output step

# RMSE Function
def rmse(predictions, targets):
    return torch.sqrt(F.mse_loss(predictions, targets))

# Load Data
file_path = 'hurdat2.txt'
hurricanes_data = parse_hurricane_data(file_path, min_length=24)

# Create Sliding Windows
window_length = 4
predict_steps = 1
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

# Training Function
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
            epoch_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / total_samples:.4f}")

# Initialize Model
input_dim = 1
lstm_output_dim = 64
epochs = 100
learning_rate = 5e-3

simple_lstm = SimpleLSTM(input_dim, lstm_output_dim, predict_steps)
optimizer = torch.optim.Adam(simple_lstm.parameters(), lr=learning_rate)

# Train Model
print("Training Simple LSTM:")
train_baseline(simple_lstm, optimizer, train_loader, epochs)

# Evaluation
all_truths = []
all_predictions = []

with torch.no_grad():
    simple_lstm.eval()
    for X_batch, y_batch in test_loader:
        predictions = simple_lstm(X_batch)
        all_truths.append(y_batch)
        all_predictions.append(predictions)

# Concatenate all results
all_truths = torch.cat(all_truths, dim=0)
all_predictions = torch.cat(all_predictions, dim=0)

# Calculate RMSE
average_rmse = rmse(all_predictions, all_truths).item()

# Save Results to CSV
with open('predictions_vs_truth.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Truth", "Predictions"])
    for truth, prediction in zip(all_truths.tolist(), all_predictions.tolist()):
        writer.writerow([truth[0], prediction[0]])

print(f"Simple LSTM Test RMSE: {average_rmse:.4f}")
