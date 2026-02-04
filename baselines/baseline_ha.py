# baseline_ha.py
import argparse
import yaml
import numpy as np
import torch

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

@torch.no_grad()
def historical_average_predict(x, horizon, num_nodes, input_dim, output_dim, ma_window=None):
    """
    x: (seq_len, batch, num_nodes * input_dim) — chuỗi quan sát gần nhất
    Trả về:
    y_hat: (horizon, batch, num_nodes * output_dim)
    Chiến lược: Moving Average trên chiều seq_len (hoặc cửa sổ ma_window cuối).
    """
    seq_len, batch_size, flat_dim = x.shape
    assert flat_dim == num_nodes * input_dim

    if ma_window is None or ma_window <= 0 or ma_window > seq_len:
        ma_window = seq_len

    # chọn cửa sổ cuối cùng
    x_win = x[-ma_window:]  # (ma_window, batch, flat_dim)

    # trung bình theo trục thời gian
    mean_last = x_win.mean(dim=0)  # (batch, num_nodes * input_dim)

    # chỉ lấy đúng số chiều output
    mean_last = mean_last[:, :num_nodes * output_dim]  # (batch, num_nodes * output_dim)

    # lặp lại cho toàn bộ horizon
    y_hat = mean_last.unsqueeze(0).repeat(horizon, 1, 1)  # (horizon, batch, num_nodes * output_dim)
    return y_hat

def main(args):
    with open(args.config_filename) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    data_cfg = cfg.get('data')
    model_cfg = cfg.get('model')

    # nạp dữ liệu & scaler
    _data = load_dataset(**data_cfg)
    scaler = _data['scaler']

    seq_len   = model_cfg['seq_len']
    horizon   = model_cfg['horizon']
    num_nodes = model_cfg['num_nodes']
    input_dim = model_cfg['input_dim']
    output_dim= model_cfg['output_dim']

    # --- Evaluate on TEST set ---
    test_iterator = _data['test_loader'].get_iterator()

    mae_list, pot_list = [], []

    for x_np, y_np in test_iterator:
        # x: (seq_len, batch, num_nodes*input_dim), y: (horizon, batch, num_nodes*output_dim)
        x, y = _prepare_data(x_np, y_np, seq_len, data_cfg['batch_size'],
                             num_nodes, input_dim, horizon, output_dim)

        # HA prediction (moving average)
        y_pred = historical_average_predict(
            x, horizon=horizon, num_nodes=num_nodes,
            input_dim=input_dim, output_dim=output_dim,
            ma_window=args.ma_window
        )

        # Tính loss trên không gian gốc (inverse transform)
        mae = _compute_loss(y, y_pred, scaler).item()
        pot = _compute_pot_loss(y, y_pred, scaler, _data['threshold']).item()

        mae_list.append(mae)
        pot_list.append(pot)

    print(f"[HA] Test MAE: {np.mean(mae_list):.4f}")
    print(f"[HA] Test Peak Over Threshold Loss: {np.mean(pot_list):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, required=True,
                        help='YAML config dùng chung với các model khác (chứa data/model/train).')
    parser.add_argument('--ma_window', type=int, default=0,
                        help='Cửa sổ moving average (0 hoặc >seq_len sẽ mặc định dùng full seq_len).')
    args = parser.parse_args()
    main(args)
