import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F

X_train_max, y_train_max, y_val_max, X_test_max, y_test_max = torch.load("data_split.pt", map_location="cpu")

print("Train X shape:", X_train_max.shape)  # Expected shape: [2944, 16, 1]
print("Train y shape:", y_train_max.shape)  # Expected shape: [2944, 1]
print("Test X shape:", X_test_max.shape)    # Expected shape: [256, 16, 1]
print("Test y shape:", y_test_max.shape)    # Expected shape: [256, 1]

# Training configuration
input_dim = 1
output_dim = 64
latent_dim = 32
future_steps = 1
epochs = 100
learning_rate = 5e-3
batch_size = 32

# Baseline 1: Simple LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, lstm_output_dim, future_steps):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_output_dim, batch_first=True)
        self.fc = nn.Linear(lstm_output_dim, future_steps)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

model = SimpleLSTM(input_dim=input_dim, lstm_output_dim=output_dim, future_steps=future_steps)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader = DataLoader(TensorDataset(X_train_max, y_train_max), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_max, y_test_max), batch_size=batch_size, shuffle=False)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = F.mse_loss(predictions, y_batch)
        loss.backward() 
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

model.eval()
all_truths = []
all_predictions = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        predictions = model(X_batch)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_truths.extend(y_batch.cpu().numpy())

rmse_score = np.sqrt(mean_squared_error(all_truths, all_predictions))
print(f"Test RMSE: {rmse_score:.4f}")

results_df = pd.DataFrame({
    'Truth': np.array(all_truths).flatten(),
    'Predictions': np.array(all_predictions).flatten()
})
results_df.to_csv('predictions_vs_truth_simple_lstm.csv', index=False)
print("Results saved to 'predictions_vs_truth_simple_lstm.csv'")
