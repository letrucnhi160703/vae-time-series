import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from vae_mixture_graph_model import VAE
import csv
import argparse
import yaml
from utils import load_graph_data
from utils import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def _compute_loss(y_true, y_predicted, standard_scaler):
    y_true = standard_scaler.inverse_transform(y_true)
    y_predicted = standard_scaler.inverse_transform(y_predicted)
    return masked_mae_loss(y_predicted, y_true)

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def main(args):
    with open(args.config_filename) as f:
        # supervisor_config = yaml.load(f)
        supervisor_config = yaml.load(f, Loader=yaml.FullLoader)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        # if args.use_cpu_only:
        #     tf.config.set_visible_devices([], 'GPU')
        # else:
        #     physical_devices = tf.config.list_physical_devices('GPU')
        #     if physical_devices:
        #         tf.config.set_visible_devices(physical_devices[0], 'GPU')
        #         tf.config.experimental.set_memory_growth(physical_devices[0], True)

        print("Supervisor config:", supervisor_config)

        _data_kwargs = supervisor_config.get('data')
        _model_kwargs = supervisor_config.get('model')
        _train_kwargs = supervisor_config.get('train')

        _data = load_dataset(**_data_kwargs)
        standard_scaler = _data['scaler']

        vae = VAE(adj_mx=adj_mx, latent_dim=args.latent_dim, num_nodes=_model_kwargs.get('num_nodes'), future_steps=args.predict_steps, 
                  use_d_knn=args.use_d_knn, use_gpd=args.use_gpd, use_bernoulli=args.use_bernoulli, 
                  **supervisor_config).to(device)

        train_iterator = _data['train_loader'].get_iterator()

        optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

        epoch_num = _train_kwargs.get('epochs', 0)
        # epoch_num = 20
        num_batches = _data['train_loader'].num_batch
        batches_seen = num_batches * epoch_num

        for epoch in range(epoch_num):
            vae.train()
            epoch_loss = 0
            total_samples = 0

            train_iterator = _data['train_loader'].get_iterator()
            
            # print('Epoch: ', epoch)
            count = 0
            for _, (x, y) in enumerate(train_iterator):
                # print("##################", count)
                optimizer.zero_grad()

                x, y = _prepare_data(x, y, _model_kwargs['seq_len'], _data_kwargs['batch_size'],
                                    _model_kwargs['num_nodes'], _model_kwargs['input_dim'], 
                                    _model_kwargs['horizon'], _model_kwargs['output_dim'])
                # print(x.shape, y.shape)

                forecasting, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero = vae(x, y, x, batches_seen)
                # print("Forecasting: ", forecasting.shape)

                if batches_seen == 0:
                    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

                loss = vae.loss_function(forecasting, y, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero)

                batches_seen += 1

                epoch_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
                # count +=1	
                # if count == 11:
                #     break								

                loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss / total_samples:.4f}")
        test_iterator = _data['test_loader'].get_iterator()
        all_truths = []
        all_predictions = []
        losses = []

        vae.eval()
        
        count = 0
        for x, y in test_iterator:
            x, y = _prepare_data(x, y, _model_kwargs['seq_len'], _data_kwargs['batch_size'],
                                _model_kwargs['num_nodes'], _model_kwargs['input_dim'],
                                _model_kwargs['horizon'], _model_kwargs['output_dim'])
            predictions, _, _, _, _, _ = vae(x, None, x, None)

            loss = _compute_loss(y, predictions, standard_scaler)
            losses.append(loss.item())

            # print("Prediction: ", predictions.shape)
            all_truths.append(y)
            all_predictions.append(predictions)

            # count +=1	
            # if count == 11:
            #     break

        # all_truths = torch.cat(all_truths, dim=0)
        # all_predictions.append(predictions)

        average_mae = np.mean(losses)

        # with open('predictions_vs_truth.csv', mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(["Truth", "Predictions"])
        #     for truth, prediction in zip(all_truths, all_predictions):
        #         writer.writerow([truth.item(), prediction.item()])

        print(f"Test MAE: {average_mae:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')

    parser.add_argument("--file_path", type=str, default='../../datasets/LD2011_2014_less.csv', help="Path to the dataset file")
    parser.add_argument("--window_length", type=int, default=12, help="Window length for sliding window")
    parser.add_argument("--predict_steps", type=int, default=1, help="Number of steps to predict")
    parser.add_argument("--percentile", type=int, default=90, help="Percentile for threshold")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--output_dim", type=int, default=64, help="Output layer dimension")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    # parser.add_argument("--lstm_output_dim", type=int, default=64, help="LSTM output dimension")
    # parser.add_argument("--gcn_output_dim", type=int, default=64, help="GCN output dimension")
    # parser.add_argument("--beta", type=float, default=0.001, help="Beta coefficient for KL loss")
    parser.add_argument("--use_gcn", action="store_true", help="Enable GCN module")
    parser.add_argument("--use_gpd", action="store_true", help="Enable GPD reparameterization")
    parser.add_argument("--use_bernoulli", action="store_true", help="Enable Bernoulli reparameterization")
    parser.add_argument("--use_d_knn", action="store_true", help="Enable differentiable KNN")
    args = parser.parse_args()
    main(args)