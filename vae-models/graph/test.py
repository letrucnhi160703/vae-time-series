import torch
import numpy as np
import yaml
from utils import load_graph_data, load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Metrics ====
def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()

def _compute_loss(y_true, y_predicted, standard_scaler=None):
    if standard_scaler:
        y_true = standard_scaler.inverse_transform(y_true)
        y_predicted = standard_scaler.inverse_transform(y_predicted)
    return masked_mae_loss(y_predicted, y_true)

def _compute_pot_loss(y_true, y_predicted, standard_scaler=None, threshold=None):
    if standard_scaler:
        y_true = standard_scaler.inverse_transform(y_true)
        y_predicted = standard_scaler.inverse_transform(y_predicted)
        threshold = standard_scaler.inverse_transform(np.array([[threshold]]))[0][0]

    mask = ((y_true != 0) & (y_true >= threshold)).float()
    mask /= mask.mean()
    loss = torch.abs(y_predicted - y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()

# ---- New metrics ----
def _compute_rmse(y_true, y_predicted, standard_scaler=None):
    if standard_scaler:
        y_true = standard_scaler.inverse_transform(y_true)
        y_predicted = standard_scaler.inverse_transform(y_predicted)
    mse = torch.mean((y_predicted - y_true) ** 2)
    return torch.sqrt(mse)

def _compute_mape(y_true, y_predicted, standard_scaler=None):
    if standard_scaler:
        y_true = standard_scaler.inverse_transform(y_true)
        y_predicted = standard_scaler.inverse_transform(y_predicted)
    mask = (y_true != 0)
    mape = torch.mean(torch.abs((y_predicted[mask] - y_true[mask]) / y_true[mask])) * 100
    return mape

# ==== Data prep ====
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

# ==== Load model ====
vae = torch.load("dknn_miss10_thres95.pt", map_location=device, weights_only=False)
vae.eval()

use_d_knn = False

f = open('../../datasets/model/dcrnn_la.yaml')
supervisor_config = yaml.load(f, Loader=yaml.FullLoader)

graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

_data_kwargs = supervisor_config.get('data')
_model_kwargs = supervisor_config.get('model')
_train_kwargs = supervisor_config.get('train')

_data = load_dataset(**_data_kwargs)
standard_scaler = _data['scaler']

X_database = torch.from_numpy(_data['x_train']).float().permute(1, 0, 2, 3).to(device)  # (seq_len, N, num_nodes)
X_database = X_database.reshape(X_database.shape[0], X_database.shape[1], X_database.shape[2] * X_database.shape[3])  # (seq_len, N, num_nodes * input_dim)

print("X_database shape: ", X_database.shape)

percent_missing = 10

num_missing_nodes = int(X_database.shape[2] * percent_missing / 100)
missing_nodes = np.random.choice(np.arange(X_database.shape[2]), size=num_missing_nodes, replace=False)

num_available = X_database.shape[2] - num_missing_nodes

test_iterator = _data['test_loader'].get_iterator()
losses, pot_losses, rmses, mapes = [], [], [], []

y_truths, y_preds = [], []

vae.eval()
count = 0
for x, y in test_iterator:
    x, y = _prepare_data(x, y, _model_kwargs['seq_len'], _data_kwargs['batch_size'],
                        _model_kwargs['num_nodes'], _model_kwargs['input_dim'],
                        _model_kwargs['horizon'], _model_kwargs['output_dim'])
            
    X_imputed = x.clone()

    if use_d_knn:
                seq_len, batch_size, num_nodes = x.shape

                X_mask = torch.zeros((x.shape[0], x.shape[1], x.shape[2]), dtype=bool).to(device)
                for node in missing_nodes:
                    X_mask[:, :, node] = True  # Mark the missing nodes in the mask
                x_missing = x.clone().to(device)
                x_missing[X_mask] = 0

                X_imputed = x_missing.clone()  .to(device)          

                for t in range(seq_len):
                    for b in range(batch_size):
                        for node_idx in range(num_nodes):
                            if X_mask[t, b, node_idx]:
                                available_nodes = (~X_mask[t, b, :]).nonzero(as_tuple=False).squeeze().to(device)
                                available_nodes = available_nodes[available_nodes != node_idx]
                                if len(available_nodes) == 0:
                                    continue

                                x_miss = x_missing[t, b, available_nodes]
                                x_full = X_database[:, b, available_nodes]

                                distances = vae.d_knn.compute_distances(x_full, x_miss).to(device)
                                weights = vae.d_knn.soft_knn(distances, x_miss).to(device)
                                target_vals = X_database[:, b, node_idx]

                                x_imputed_val = torch.sum(weights * target_vals) / torch.sum(weights)
                                X_imputed[t, b, node_idx] = x_imputed_val

    predictions, _, _, _, _, _, xi_extreme_pred, sigma_extreme_pred = vae(x=X_imputed, y=None, batches_seen=None)

    # Compute metrics
    loss = _compute_loss(y, predictions, standard_scaler)
    losses.append(loss.item())

    pot_loss = _compute_pot_loss(y, predictions, standard_scaler, _data['threshold'])
    pot_losses.append(pot_loss.item())

    rmse = _compute_rmse(y, predictions, standard_scaler)
    rmses.append(rmse.item())

    mape = _compute_mape(y, predictions, standard_scaler)
    mapes.append(mape.item())

    y_true = y
    y_predicted = predictions
    y_truths.append(y_true.detach().cpu())
    y_preds.append(y_predicted.detach().cpu())

    count += 1
    if count == 101:
        break

# ==== Summary ====
average_mae = np.mean(losses)
average_pot_loss = np.mean(pot_losses)
average_rmse = np.mean(rmses)
average_mape = np.mean(mapes)

# y_preds = np.concatenate(y_preds, axis=1)
# y_truths = np.concatenate(y_truths, axis=1)

# node_idx = 11
# y_pred_series = y_preds[:, :, node_idx].reshape(-1)
# y_truth_series = y_truths[:, :, node_idx].reshape(-1)

# data = np.stack([y_truth_series, y_pred_series], axis=1)
# np.savetxt("prediction_vs_truth.csv", data, delimiter=",", header="Truth,Predictions", comments='')

print(f"Test MAE: {average_mae:.4f}")
print(f"Test RMSE: {average_rmse:.4f}")
print(f"Test MAPE: {average_mape:.4f}%")
print(f"Test Peak Over Threshold Loss: {average_pot_loss:.4f}")
