import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Simple neural network model for DKNN imputation
# Learning the temperature parameter t in Soft k-NN
class DKNNImputer3D(nn.Module):
    def __init__(self, input_dim):
        super(DKNNImputer3D, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.t = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Compute Euclidean distances between the missing values and the full values
def compute_distances(X, x_new):
    return torch.norm(X - x_new, dim=1)

# Compute weights using soft k-NN with an exponential function
def soft_knn(distances, t):
    weights = torch.exp(-distances / t)
    return weights

# Compute Mean Absolute Error (MAE) loss
def compute_mae(X_imputed, X_full, mask):
    return torch.mean(torch.abs(X_imputed[mask] - X_full[mask]))

# seq_length: number of time steps of X_test_missing
# batch_size: number of samples in a batch
# num_nodes: number of nodes in the graph
seq_length, batch_size, num_nodes, percent_missing = 3, 2, 3, 35

# X_database is the complete historical data, with many time steps
X_database = np.random.rand(1000, batch_size, num_nodes)
X_database_tensor = torch.tensor(X_database, dtype=torch.float32)

# Randomly select a segment to create X_train_missing.
# The shape of X_train_missing is the same as X_test_missing
start_idx = np.random.randint(0, X_database_tensor.shape[0] - seq_length)

# X_train_missing_truth is a segment of X_database_tensor with the same shape as X_test_missing. Used for training.
X_train_missing_truth = X_database_tensor[start_idx:start_idx + seq_length]
X_train_missing = X_train_missing_truth.clone()

# X_train_full is the remaining part of X_database_tensor, excluding the segment used for X_train_missing
X_train_full = torch.cat([X_database_tensor[:start_idx], X_database_tensor[start_idx + seq_length:]])

print("X_database shape:", X_database_tensor.shape)
print("X_train_missing shape:", X_train_missing.shape)
print("X_train_full shape:", X_train_full.shape)

# Randomly select nodes to be missing in X_train_missing and X_test_missing
num_missing_nodes = int(num_nodes * percent_missing / 100)
missing_nodes = np.random.choice(np.arange(num_nodes), size=num_missing_nodes, replace=False)
print("Missing nodes:", missing_nodes)

X_train_mask = torch.zeros((X_train_missing.shape[0], X_train_missing.shape[1], X_train_missing.shape[2]), dtype=bool)
for node in missing_nodes:
    X_train_mask[:, :, node] = True  # Mark the missing nodes in the mask
X_test_mask = X_train_mask.clone()  # Assuming the same missing nodes for X_test_missing

# Randomly generate X_test_truth and mask it with the same mask as X_train_missing
X_test_truth = torch.tensor(np.random.rand(seq_length, batch_size, num_nodes), dtype=torch.float32)
X_test_missing = X_test_truth.clone()
X_test_missing[X_test_mask] = 0

X_train_missing[X_train_mask] = 0

# print("X_train_missing: ", X_train_missing)
# print("X_train_mask:", X_train_mask)
# print("X_test_missing: ", X_test_missing)
# print("X_test_mask:", X_test_mask)


# Initialize the model and optimizer
model = DKNNImputer3D(input_dim=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    X_imputed = X_train_missing.clone()  # A copy of X_train_missing to update
    # print("Initial X_imputed:", X_imputed)
    for t in range(seq_length):
        for b in range(batch_size):
            for node_idx in range(num_nodes):
                if X_train_mask[t, b, node_idx]:
                    available_nodes = (~X_train_mask[t, b, :]).nonzero(as_tuple=False).squeeze()
                    available_nodes = available_nodes[available_nodes != node_idx]
                    if len(available_nodes) == 0:
                        continue

                    # print("Avaliable nodes:", available_nodes)

                    x_miss = X_train_missing[t, b, available_nodes]
                    x_full = X_train_full[:, b, available_nodes]

                    distances = compute_distances(x_full, x_miss)
                    weights = soft_knn(distances, model.t)
                    target_vals = X_train_full[:, b, node_idx]

                    x_imputed_val = torch.sum(weights * target_vals) / torch.sum(weights)
                    # print('x_imputed_val:', x_imputed_val)
                    X_imputed[t, b, node_idx] = x_imputed_val
                    # print("Updated X_imputed:", X_imputed)

    # Calculate MAE loss between X_imputed and X_train_missing_truth
    loss = compute_mae(X_imputed, X_train_missing_truth, X_train_mask)

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
# print("X_train_missing_truth:", X_train_missing_truth)

# Testing loop
model.eval()
with torch.no_grad():
    X_test_imputed = X_test_missing.clone()
    for t in range(seq_length):
        for b in range(batch_size):
            for node_idx in range(num_nodes):
                if X_test_mask[t, b, node_idx]:
                    available_nodes = (~X_test_mask[t, b, :]).nonzero(as_tuple=False).squeeze()
                    available_nodes = available_nodes[available_nodes != node_idx]
                    if len(available_nodes) == 0:
                        continue

                    # print("Avaliable nodes:", available_nodes)

                    x_miss = X_test_missing[t, b, available_nodes]
                    x_full = X_database_tensor[:, b, available_nodes]

                    distances = compute_distances(x_full, x_miss)
                    weights = soft_knn(distances, model.t)
                    target_vals = X_database_tensor[:, b, node_idx]

                    x_imputed_val = torch.sum(weights * target_vals) / torch.sum(weights)
                    # print('x_imputed_val:', x_imputed_val)
                    X_test_imputed[t, b, node_idx] = x_imputed_val
                    # print("Updated X_imputed:", X_test_imputed)


mae_loss = compute_mae(X_test_imputed, X_test_truth, X_test_mask)
print(f"MAE Loss on Test Data: {mae_loss.item():.4f}")
