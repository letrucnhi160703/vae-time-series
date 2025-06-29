import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import yaml
from utils import load_dataset
import csv

# ====== Model with Embedding and DKNN logic ======
import torch
import torch.nn as nn

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

# ====== Utility functions ======

def compute_mae(X_imputed, X_full, mask):
    return torch.mean(torch.abs(X_imputed[mask] - X_full[mask]))

def _get_x_y(x, y):
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    x = x.permute(1, 0, 2, 3)
    y = y.permute(1, 0, 2, 3)
    return x, y

def _get_x_y_in_correct_dims(x, y, seq_len, batch_size, num_nodes, input_dim, horizon, output_dim):
    x = x.view(seq_len, batch_size, num_nodes * input_dim)
    y = y[..., :output_dim].view(horizon, batch_size, num_nodes * output_dim)
    return x, y

def _prepare_data(x, y, seq_len, batch_size, num_nodes, input_dim, horizon, output_dim):
    x, y = _get_x_y(x, y)
    x, y = _get_x_y_in_correct_dims(x, y, seq_len, batch_size, num_nodes, input_dim, horizon, output_dim)
    return x.to(device), y.to(device)

# ====== Main training loop ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    _data_kwargs = config['data']
    _model_kwargs = config['model']

    _data = load_dataset(**_data_kwargs)
    train_loader = _data['train_loader'].get_iterator()

    input_dim = _data['x_train'].shape[-1]
    print("_data['x_train'].shape[-1]: ", _data['x_train'].shape[-1])
    num_nodes = _model_kwargs['num_nodes']

     # Dùng toàn bộ x_train làm database để tìm nearest neighbors
    X_database = torch.from_numpy(_data['x_train']).float().permute(1, 0, 2, 3)  # (seq_len, N, num_nodes)
    X_database = X_database.reshape(X_database.shape[0], X_database.shape[1], X_database.shape[2] * X_database.shape[3])  # (seq_len, N, num_nodes * input_dim)

    print("X_database shape: ", X_database.shape)
    
    percent_missing = 35  # % nodes bị missing

    # Tạo mask (ngẫu nhiên)
    num_missing_nodes = int(X_database.shape[2] * percent_missing / 100)
    # print("num_missing_nodes: ", num_missing_nodes)
    missing_nodes = np.random.choice(np.arange(X_database.shape[2]), size=num_missing_nodes, replace=False)
    # print("Missing nodes:", missing_nodes)

    num_available = X_database.shape[2] - num_missing_nodes

    model = DKNNImputer3D(
        input_dim=_data['x_train'].shape[-1],
        t=1.0,
        seq_len=_model_kwargs['seq_len'],
        available_node=num_available
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-06)

    epochs = 10
    train_loader = _data['train_loader'].get_iterator()
    test_loader = _data['test_loader'].get_iterator()

    
    # all_truths = []
    # all_predictions = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        count = 0
        for _, (x, y) in enumerate(train_loader):
            x, _ = _prepare_data(x, y, _model_kwargs['seq_len'], _data_kwargs['batch_size'],
                                _model_kwargs['num_nodes'], _model_kwargs['input_dim'], 
                                _model_kwargs['horizon'], _model_kwargs['output_dim'])
            # x = torch.from_numpy(x).float().permute(1, 0, 2, 3)[..., 0]  # (seq_len, batch_size, num_nodes)
            seq_len, batch_size, num_nodes = x.shape

            # print("seq_len:", seq_len, "batch_size:", batch_size, "num_nodes:", num_nodes)

            

            X_mask = torch.zeros((x.shape[0], x.shape[1], x.shape[2]), dtype=bool)
            for node in missing_nodes:
                X_mask[:, :, node] = True  # Mark the missing nodes in the mask
            # X_test_mask = X_train_mask.clone()
            x_missing = x.clone()
            x_missing[X_mask] = 0

            X_imputed = x_missing.clone()

            # print("X_mask shape:", X_mask.shape)
            # print("x_missing shape:", x_missing.shape)
            # print("X_imputed shape:", X_imputed.shape)               

            for t in range(seq_len):
                for b in range(batch_size):
                    for node_idx in range(num_nodes):
                        if X_mask[t, b, node_idx]:
                            available_nodes = (~X_mask[t, b, :]).nonzero(as_tuple=False).squeeze()
                            available_nodes = available_nodes[available_nodes != node_idx]
                            if len(available_nodes) == 0:
                                continue
                            
                            # print("Available nodes:", len(available_nodes))

                            x_miss = x_missing[t, b, available_nodes]
                            x_full = X_database[:, b, available_nodes]

                            # print("x_miss shape:", x_miss.shape)
                            # print("x_full shape:", x_full.shape)

                            distances = model.compute_distances(x_full, x_miss)
                            weights = model.soft_knn(distances, x_miss)
                            # print("weights shape: ", weights.shape)
                            target_vals = X_database[:, b, node_idx]

                            x_imputed_val = torch.sum(weights * target_vals) / torch.sum(weights)
                            X_imputed[t, b, node_idx] = x_imputed_val
                            # all_truths.append(X_database[t, b, node_idx])
                            # all_predictions.append(x_imputed_val)

            loss = compute_mae(X_imputed, x, X_mask)

            # print("Count: ", count)
            count +=1	
            if count == 11:
                break	

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, t: {model.t.item():.4f}")
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # with open('predictions_vs_truth.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Truth", "Predictions"])
    #     for truth, prediction in zip(all_truths, all_predictions):
    #         writer.writerow([truth.item(), prediction.item()])

    model.eval()
    all_imputed = []
    all_truth = []
    all_mask = []
    count = 0
    for _, (x, y) in enumerate(test_loader):
        x, _ = _prepare_data(x, y, _model_kwargs['seq_len'], _data_kwargs['batch_size'],
                            _model_kwargs['num_nodes'], _model_kwargs['input_dim'], 
                            _model_kwargs['horizon'], _model_kwargs['output_dim'])
        seq_len, batch_size, num_nodes = x.shape

        X_mask = torch.zeros((x.shape[0], x.shape[1], x.shape[2]), dtype=bool)
        for node in missing_nodes:
            X_mask[:, :, node] = True  # Mark the missing nodes in the mask
        # X_test_mask = X_train_mask.clone()
        x_missing = x.clone()
        x_missing[X_mask] = 0

        X_imputed = x_missing.clone()

        # print("X_mask shape:", X_mask.shape)
        # print("x_missing shape:", x_missing.shape)
        # print("X_imputed shape:", X_imputed.shape)              

        for t in range(seq_len):
            for b in range(batch_size):
                for node_idx in range(num_nodes):
                    if X_mask[t, b, node_idx]:
                        available_nodes = (~X_mask[t, b, :]).nonzero(as_tuple=False).squeeze()
                        available_nodes = available_nodes[available_nodes != node_idx]
                        if len(available_nodes) == 0:
                            continue
                        
                        # print("Available nodes:", len(available_nodes))

                        x_miss = x_missing[t, b, available_nodes]
                        x_full = X_database[:, b, available_nodes]

                        # print("x_miss shape:", x_miss.shape)
                        # print("x_full shape:", x_full.shape)

                        distances = model.compute_distances(x_full, x_miss)
                        weights = model.soft_knn(distances, x_miss)
                        # print("weights shape: ", weights.shape)
                        target_vals = X_database[:, b, node_idx]

                        x_imputed_val = torch.sum(weights * target_vals) / torch.sum(weights)
                        X_imputed[t, b, node_idx] = x_imputed_val
                        # all_truths.append(X_database[t, b, node_idx])
                        # all_predictions.append(x_imputed_val)

        all_imputed.append(X_imputed)
        all_truth.append(x)
        all_mask.append(X_mask)

        # print("Count: ", count)
        count +=1	
        if count == 11:
            break

    X_imputed_full = torch.cat(all_imputed, dim=1)  # batch dim
    X_truth_full = torch.cat(all_truth, dim=1)
    X_mask_full = torch.cat(all_mask, dim=1)

    total_mae = compute_mae(X_imputed_full, X_truth_full, X_mask_full)
    print(f"Test Test MAE: {total_mae.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, required=True, help='YAML config file path')
    args = parser.parse_args()
    main(args)
