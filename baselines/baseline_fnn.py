# baseline_ha.py
import argparse
import yaml
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from utils import load_graph_data, load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
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

    # print("y_true shape: ", y_true.shape)
    # print("y_predicted shape: ", y_predicted.shape)
    # print("Threshold shape: ", threshold.shape)

    mask = ((y_true != 0) & (y_true >= threshold)).float()
    # if mask.sum() != 0:
    #     print("Has extreme values.")
    mask /= mask.mean()
    loss = torch.abs(y_predicted - y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()

def _get_x_y(x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        # self._logger.debug("X: {}".format(x.size()))
        # self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

def _get_x_y_in_correct_dims(x, y, seq_len, batch_size, num_nodes, input_dim, horizon, output_dim):
    """
    :param x: shape (seq_len, batch_size, num_sensor, input_dim)
    :param y: shape (horizon, batch_size, num_sensor, input_dim)
    :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
             y: shape (horizon, batch_size, num_sensor * output_dim)
    """
    # batch_size = x.size(1)
    x = x.view(seq_len, batch_size, num_nodes * input_dim)
    # print("##########", x.shape)
    y = y[..., :output_dim].view(horizon, batch_size,
                                    num_nodes * output_dim)
    return x, y

def _prepare_data(x, y, seq_len, batch_size, num_nodes, input_dim, horizon, output_dim):
        x, y = _get_x_y(x, y)
        x, y = _get_x_y_in_correct_dims(x, y, seq_len, batch_size, num_nodes, input_dim, horizon, output_dim)
        return x.to(device), y.to(device)

class FNN(nn.Module):
    def __init__(self, seq_len, num_nodes, input_dim, horizon, output_dim, hidden_dim=128):
        super(FNN, self).__init__()
        in_dim = seq_len * num_nodes * input_dim
        out_dim = horizon * num_nodes * output_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x: (seq_len, batch, num_nodes*input_dim)
        seq_len, batch, flat_dim = x.shape
        x = x.permute(1, 0, 2).reshape(batch, -1)  # (batch, seq_len*flat_dim)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)  # (batch, horizon*num_nodes*output_dim)
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

    fnn = FNN(seq_len, num_nodes, input_dim, horizon, output_dim,
              hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(fnn.parameters(), lr=args.lr)

    X_database = torch.from_numpy(_data['x_train']).float().permute(1, 0, 2, 3).to(device)  # (seq_len, N, num_nodes)
    X_database = X_database.reshape(X_database.shape[0], X_database.shape[1], X_database.shape[2] * X_database.shape[3])  # (seq_len, N, num_nodes * input_dim)

    print("X_database shape: ", X_database.shape)
    
    percent_missing = 40

    num_missing_nodes = int(X_database.shape[2] * percent_missing / 100)
    missing_nodes = np.random.choice(np.arange(X_database.shape[2]), size=num_missing_nodes, replace=False)

    use_d_knn = True

    # Training
    for epoch in range(args.epochs):
        fnn.train()
        train_losses = []
        count = 0
        for x_np, y_np in _data['train_loader'].get_iterator():
            x, y = _prepare_data(x_np, y_np, seq_len, data_cfg['batch_size'],
                                 num_nodes, input_dim, horizon, output_dim)

            X_imputed = x.clone().to(device)

            if use_d_knn:
                for node in missing_nodes:
                    start = node * input_dim
                    end = start + input_dim
                    X_imputed[:, :, start:end] = 0
           
            y_pred = fnn(X_imputed)

            y_pred = y_pred.view(data_cfg['batch_size'], horizon, num_nodes*output_dim).permute(1,0,2)

            loss = _compute_loss(y, y_pred, scaler)
            optimizer.zero_grad()

            count +=1	
            if count == 101:
                break
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {np.mean(train_losses):.4f}")

    # Evaluation
    fnn.eval()
    mae_list, rmse_list, mape_list, pot_list = [], [], [], []
    count = 0
    with torch.no_grad():
        for x_np, y_np in _data['test_loader'].get_iterator():
            x, y = _prepare_data(x_np, y_np, seq_len, data_cfg['batch_size'],
                                 num_nodes, input_dim, horizon, output_dim)
           
            X_imputed = x.clone().to(device)

            if use_d_knn:
                for node in missing_nodes:
                    start = node * input_dim
                    end = start + input_dim
                    X_imputed[:, :, start:end] = 0
           
            y_pred = fnn(X_imputed)
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

            count +=1	
            if count == 101:
                break

    print(f"[FNN] Test MAE: {np.mean(mae_list):.4f}")
    print(f"[FNN] Test POT Loss: {np.mean(pot_list):.4f}")
    print(f"[FNN] Test RMSE: {np.mean(rmse_list):.4f}")
    print(f"[FNN] Test MAPE: {np.mean(mape_list):.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, required=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    main(args)