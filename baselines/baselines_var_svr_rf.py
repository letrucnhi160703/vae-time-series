import argparse
import yaml
import numpy as np
import torch
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

from utils import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==== Metrics ====
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def pot_mae(y_true, y_pred, threshold):
    mask = (y_true >= threshold)
    if mask.sum() == 0:
        return 0
    return mean_absolute_error(y_true[mask], y_pred[mask])


# ==== Data helper ====
def _prepare_flat_data(data_loader, seq_len, batch_size, num_nodes, input_dim, horizon, output_dim):
    X_list, Y_list = [], []
    for x_np, y_np in data_loader.get_iterator():
        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float()
        # Flatten theo batch
        x = x.reshape(batch_size, -1).numpy()
        y = y.reshape(batch_size, -1).numpy()
        X_list.append(x)
        Y_list.append(y)
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    return X, Y


# ==== VAR baseline ====
def run_var(train_y, test_y, horizon=1):
    model = VAR(train_y)
    maxlags = min(5, len(train_y)//2 - 1)
    model_fitted = model.fit(maxlags=maxlags, ic='aic')
    preds = model_fitted.forecast(train_y[-model_fitted.k_ar:], steps=len(test_y))
    return preds


# ==== SVR baseline ====
def run_svr(X_train, Y_train, X_test):
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train, Y_train.ravel())
    return svr.predict(X_test)


# ==== Random Forest baseline ====
def run_rf(X_train, Y_train, X_test):
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, Y_train.ravel())
    return rf.predict(X_test)


# ==== Main Experiment ====
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

    # Flatten data for traditional models
    X_train, Y_train = _prepare_flat_data(_data['train_loader'], seq_len, data_cfg['batch_size'],
                                        num_nodes, input_dim, horizon, output_dim)
    X_test, Y_test = _prepare_flat_data(_data['test_loader'], seq_len, data_cfg['batch_size'],
                                        num_nodes, input_dim, horizon, output_dim)

    # ==== Chọn đầu ra để khớp với X ====
    # Cách 1: dùng trung bình các node
    Y_train = Y_train.mean(axis=1, keepdims=True)
    Y_test  = Y_test.mean(axis=1, keepdims=True)

    threshold = _data['threshold']
    Y_train_inv = scaler.inverse_transform(Y_train)
    Y_test_inv  = scaler.inverse_transform(Y_test)

    # # ========== VAR ==========
    # try:
    #     preds_var = run_var(Y_train_inv, Y_test_inv)
    #     mae_var = mae(Y_test_inv[:len(preds_var)], preds_var)
    #     rmse_var = rmse(Y_test_inv[:len(preds_var)], preds_var)
    #     mape_var = mape(Y_test_inv[:len(preds_var)], preds_var)
    #     pot_var = pot_mae(Y_test_inv[:len(preds_var)], preds_var, threshold)
    #     print(f"[VAR] MAE={mae_var:.4f}, RMSE={rmse_var:.4f}, MAPE={mape_var:.2f}%, POT={pot_var:.4f}")
    # except Exception as e:
    #     print("[VAR] skipped due to:", e)

    # ========== SVR ==========
    print("Running SVR baselines...")
    preds_svr = run_svr(X_train, Y_train, X_test)
    preds_svr_inv = scaler.inverse_transform(preds_svr.reshape(-1, 1))
    mae_svr = mae(Y_test_inv.ravel(), preds_svr_inv.ravel())
    rmse_svr = rmse(Y_test_inv.ravel(), preds_svr_inv.ravel())
    mape_svr = mape(Y_test_inv.ravel(), preds_svr_inv.ravel())
    pot_svr = pot_mae(Y_test_inv.ravel(), preds_svr_inv.ravel(), threshold)
    print(f"[SVR] MAE={mae_svr:.4f}, RMSE={rmse_svr:.4f}, MAPE={mape_svr:.2f}%, POT={pot_svr:.4f}")

    # ========== Random Forest ==========
    print("Running Random Forest baselines...")
    preds_rf = run_rf(X_train, Y_train, X_test)
    preds_rf_inv = scaler.inverse_transform(preds_rf.reshape(-1, 1))
    mae_rf = mae(Y_test_inv.ravel(), preds_rf_inv.ravel())
    rmse_rf = rmse(Y_test_inv.ravel(), preds_rf_inv.ravel())
    mape_rf = mape(Y_test_inv.ravel(), preds_rf_inv.ravel())
    pot_rf = pot_mae(Y_test_inv.ravel(), preds_rf_inv.ravel(), threshold)
    print(f"[RF] MAE={mae_rf:.4f}, RMSE={rmse_rf:.4f}, MAPE={mape_rf:.2f}%, POT={pot_rf:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, required=True)
    args = parser.parse_args()
    main(args)
