import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def _get_x_y(x, y):
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    x = x.permute(1, 0, 2, 3)  # (seq_len, batch, num_nodes, input_dim)
    y = y.permute(1, 0, 2, 3)  # (horizon, batch, num_nodes, input_dim)
    return x, y

def _get_x_y_in_correct_dims(x, y, seq_len, batch_size, num_nodes, input_dim, horizon, output_dim):
    x = x.view(seq_len, batch_size, num_nodes * input_dim)
    y = y[..., :output_dim].view(horizon, batch_size, num_nodes * output_dim)
    return x, y

def _prepare_data(x, y, seq_len, batch_size, num_nodes, input_dim, horizon, output_dim):
    x, y = _get_x_y(x, y)
    x, y = _get_x_y_in_correct_dims(x, y, seq_len, batch_size, num_nodes, input_dim, horizon, output_dim)
    return x.to(device), y.to(device)

class FCLSTM(nn.Module):
    def __init__(self, seq_len, num_nodes, input_dim, horizon, output_dim,
                 hidden_dim=128, num_layers=1):
        super(FCLSTM, self).__init__()
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.horizon = horizon
        self.output_dim = output_dim

        in_dim = num_nodes * input_dim
        out_dim = num_nodes * output_dim

        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=False)  # input: (seq_len, batch, in_dim)

        self.fc = nn.Linear(hidden_dim, out_dim * horizon)

    def forward(self, x):
        # x: (seq_len, batch, num_nodes*input_dim)
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch, hidden_dim)
        h_last = h_n[-1]            # (batch, hidden_dim)
        out = self.fc(h_last)       # (batch, horizon*num_nodes*output_dim)
        return out

def main(args):
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

    model = FCLSTM(seq_len, num_nodes, input_dim, horizon, output_dim,
                   hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    X_database = torch.from_numpy(_data['x_train']).float().permute(1, 0, 2, 3).to(device)  # (seq_len, N, num_nodes)
    X_database = X_database.reshape(X_database.shape[0], X_database.shape[1], X_database.shape[2] * X_database.shape[3])  # (seq_len, N, num_nodes * input_dim)

    print("X_database shape: ", X_database.shape)
    
    percent_missing = 40

    num_missing_nodes = int(X_database.shape[2] * percent_missing / 100)
    missing_nodes = np.random.choice(np.arange(X_database.shape[2]), size=num_missing_nodes, replace=False)
    
    use_d_knn = True

    # Training
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        batch_count = 0
        for x_np, y_np in _data['train_loader'].get_iterator():
            x, y = _prepare_data(x_np, y_np, seq_len, data_cfg['batch_size'],
                                 num_nodes, input_dim, horizon, output_dim)

            X_imputed = x.clone().to(device)

            if use_d_knn:
                for node in missing_nodes:
                    start = node * input_dim
                    end = start + input_dim
                    X_imputed[:, :, start:end] = 0
           
            y_pred = model(X_imputed)
            
            y_pred = y_pred.view(data_cfg['batch_size'], horizon, num_nodes*output_dim).permute(1,0,2)

            loss = _compute_loss(y, y_pred, scaler)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            batch_count += 1
            if batch_count >= args.max_train_batches:
                break
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {np.mean(train_losses):.4f}")

    # Evaluation
    model.eval()
    mae_list, rmse_list, mape_list, pot_list = [], [], [], []
    with torch.no_grad():
        test_count = 0
        for x_np, y_np in _data['test_loader'].get_iterator():
            x, y = _prepare_data(x_np, y_np, seq_len, data_cfg['batch_size'],
                                 num_nodes, input_dim, horizon, output_dim)

            X_imputed = x.clone().to(device)

            if use_d_knn:
                for node in missing_nodes:
                    start = node * input_dim
                    end = start + input_dim
                    X_imputed[:, :, start:end] = 0
            y_pred = model(X_imputed)
            
            y_pred = y_pred.view(data_cfg['batch_size'], horizon, num_nodes*output_dim).permute(1,0,2)

            mae = _compute_loss(y, y_pred, scaler).item()
            pot = _compute_pot_loss(y, y_pred, scaler, _data['threshold']).item()
            rmse = torch.sqrt(torch.mean((y_pred - y)**2)).item()
            mask = y != 0
            mape = torch.mean(torch.abs((y_pred[mask] - y[mask]) / y[mask])) * 100
            mae_list.append(mae)
            pot_list.append(pot)
            rmse_list.append(rmse)
            mape_list.append(mape)

            test_count += 1
            if test_count >= args.max_test_batches:
                break

    print(f"[FC-LSTM] Test MAE: {np.mean(mae_list):.4f}")
    print(f"[FC-LSTM] Test POT Loss: {np.mean(pot_list):.4f}")
    print(f"[FC-LSTM] Test RMSE: {np.mean(rmse_list):.4f}")
    print(f"[FC-LSTM] Test MAPE: {np.mean(mape_list):.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, required=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max_train_batches', type=int, default=100,
                        help="Chỉ train trên N batch đầu")
    parser.add_argument('--max_test_batches', type=int, default=10,
                        help="Chỉ test trên N batch đầu")
    args = parser.parse_args()
    main(args)
