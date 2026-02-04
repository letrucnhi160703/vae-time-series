import argparse
import yaml
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from utils import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Loss & Metrics ====
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


# ==== Data reshape ====
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


# ==== Models (multi-step output) ====
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, horizon, num_layers=1):
        super(RNNModel, self).__init__()
        self.horizon = horizon
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim * horizon)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])                # (batch, horizon*output_dim)
        out = out.view(x.size(0), self.horizon, -1) # (batch, horizon, output_dim)
        return out


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, horizon, num_layers=1):
        super(GRUModel, self).__init__()
        self.horizon = horizon
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim * horizon)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])                # (batch, horizon*output_dim)
        out = out.view(x.size(0), self.horizon, -1) # (batch, horizon, output_dim)
        return out


# ==== Main ====
def train_and_eval(model_name, args):
    with open(args.config_filename) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    data_cfg = cfg.get('data')
    model_cfg = cfg.get('model')

    _data = load_dataset(**data_cfg)
    scaler = _data['scaler']

    seq_len   = model_cfg['seq_len']
    horizon   = model_cfg['horizon']
    num_nodes = model_cfg['num_nodes']
    input_dim = model_cfg['input_dim']
    output_dim= model_cfg['output_dim']

    input_size  = num_nodes * input_dim
    output_size = num_nodes * output_dim

    if model_name == "RNN":
        model = RNNModel(input_size, args.hidden_dim, output_size, horizon).to(device)
    else:
        model = GRUModel(input_size, args.hidden_dim, output_size, horizon).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    X_database = torch.from_numpy(_data['x_train']).float().permute(1, 0, 2, 3).to(device)  # (seq_len, N, num_nodes)
    X_database = X_database.reshape(X_database.shape[0], X_database.shape[1], X_database.shape[2] * X_database.shape[3])  # (seq_len, N, num_nodes * input_dim)

    print("X_database shape: ", X_database.shape)
    
    percent_missing = 40

    num_missing_nodes = int(X_database.shape[2] * percent_missing / 100)
    missing_nodes = np.random.choice(np.arange(X_database.shape[2]), size=num_missing_nodes, replace=False)

    use_d_knn = True

    # ==== Training ====
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        count = 0
        for x_np, y_np in _data['train_loader'].get_iterator():
            x, y = _prepare_data(x_np, y_np, seq_len, data_cfg['batch_size'],
                                 num_nodes, input_dim, horizon, output_dim)
            x = x.permute(1, 0, 2)  # (batch, seq_len, input_size)
            y = y.permute(1, 0, 2)  # (batch, horizon, output_size)

            X_imputed = x.clone().to(device)

            if use_d_knn:
                for node in missing_nodes:
                    start = node * input_dim
                    end = start + input_dim
                    X_imputed[:, :, start:end] = 0

            optimizer.zero_grad()
            y_pred = model(X_imputed)       # (batch, horizon, output_size)
            loss = _compute_loss(y, y_pred, scaler)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            count += 1
            if count > 101:
                break

        print(f"[{model_name}] Epoch {epoch+1}/{args.epochs}, Train Loss: {np.mean(train_losses):.4f}")

    # ==== Evaluation ====
    model.eval()
    mae_list, rmse_list, mape_list, pot_list = [], [], [], []
    count = 0
    with torch.no_grad():
        for x_np, y_np in _data['test_loader'].get_iterator():
            x, y = _prepare_data(x_np, y_np, seq_len, data_cfg['batch_size'],
                                 num_nodes, input_dim, horizon, output_dim)
            x = x.permute(1, 0, 2)
            y = y.permute(1, 0, 2)

            X_imputed = x.clone().to(device)

            if use_d_knn:
                for node in missing_nodes:
                    start = node * input_dim
                    end = start + input_dim
                    X_imputed[:, :, start:end] = 0

            y_pred = model(X_imputed)
            y_true = y
            y_hat  = y_pred

            if scaler:
                y_true = scaler.inverse_transform(y)
                y_hat  = scaler.inverse_transform(y_pred)

            # compute metrics
            mae = torch.mean(torch.abs(y_hat - y_true)).item()
            rmse = torch.sqrt(torch.mean((y_hat - y_true)**2)).item()
            mask = y_true != 0
            mape = torch.mean(torch.abs((y_hat[mask] - y_true[mask]) / y_true[mask])) * 100
            pot = _compute_pot_loss(y, y_pred, scaler, _data['threshold']).item()

            mae_list.append(mae)
            rmse_list.append(rmse)
            mape_list.append(mape)
            pot_list.append(pot)

            count += 1
            if count > 101:
                break

    print(f"[{model_name}] Test MAE: {np.mean(mae_list):.4f}, RMSE: {np.mean(rmse_list):.4f}, "
          f"MAPE: {np.mean(mape_list):.2f}%, POT MAE: {np.mean(pot_list):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, required=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, choices=['RNN', 'GRU'], default='RNN')
    args = parser.parse_args()

    train_and_eval(args.model, args)
