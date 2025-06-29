import torch
import torch.nn as nn

# Simple neural network model for DKNN imputation
# Learning the temperature parameter t in Soft k-NN
class DKNNImputer3D(nn.Module):
    def __init__(self, input_dim, t=1.0, seq_len=None, available_node=100):
        """
        DKNN Imputer:
        - learnable temperature t
        - dynamic alpha via MLP
        """
        super(DKNNImputer3D, self).__init__()
        self.t = nn.Parameter(torch.tensor(t, dtype=torch.float32))

        assert seq_len is not None, "Bạn phải truyền seq_len!"

        # MLP để sinh alpha động từ x_miss
        self.mlp = nn.Sequential(
            nn.Linear(available_node, 32),
            nn.ReLU(),
            nn.Linear(32, seq_len),
            nn.Sigmoid()  # scale alpha ∈ (0,1)
        )

    def compute_distances(self, X, x_new):
        return torch.norm(X - x_new, dim=1)

    def soft_knn(self, distances, x_miss):
        """
        distances: [seq_len]
        x_miss: [num_neighbors]
        """
        base_weights = torch.exp(-distances / self.t)
        alpha = self.mlp(x_miss)  # [seq_len]
        weights = base_weights * alpha
        return weights

# Compute Mean Absolute Error (MAE) loss
def compute_mae(X_imputed, X_full, mask):
    return torch.mean(torch.abs(X_imputed[mask] - X_full[mask]))