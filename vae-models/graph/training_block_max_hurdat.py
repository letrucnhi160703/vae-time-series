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
import scipy.io
import os, random
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE model with different hyperparameters.")
    
    # parser.add_argument("--file_path", type=str, default='../../datasets/LD2011_2014_less.csv', help="Path to the dataset file")
    # parser.add_argument("--window_length", type=int, default=12, help="Window length for sliding window")
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

    return parser.parse_args()

def rmse(predictions, truths):
    return ((truths - predictions) ** 2).mean().sqrt().item()


def visualize_distribution(y_train_max, y_test_max):
    # Tạo một biểu đồ phân phối cho y_train_max và y_test_max
    plt.figure(figsize=(12, 6))

    # Biểu đồ histogram cho y_train_max
    plt.subplot(1, 2, 1)
    plt.hist(y_train_max.numpy(), bins=30, color='blue', alpha=0.7, label='y_train_max')
    plt.title('Distribution of y_train_max')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Biểu đồ histogram cho y_test_max
    plt.subplot(1, 2, 2)
    plt.hist(y_test_max.numpy(), bins=30, color='red', alpha=0.7, label='y_test_max')
    plt.title('Distribution of y_test_max')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Hiển thị các biểu đồ
    plt.tight_layout()
    plt.show()


def extend_last_batch(X, y, batch_size=64):
    last_batch_size = X.shape[0] % batch_size
    if last_batch_size != 0:
        indices = [i for i in range(0, (y.shape[0]-last_batch_size))]

        index = random.sample(indices, batch_size - last_batch_size)
        X_extended = X[index]
        y_extended = y[index]
        X = torch.cat((X, X_extended), 0)
        y = torch.cat((y, y_extended), 0)
    return X, y

def create_X_data(dataset, time_step=1):
    dataX = []
    for i in range(len(dataset)):
        X_data = dataset[i][0:time_step]
        dataX.append(X_data)
    return np.array(dataX)

def ready_X_data(train_data, val_data, test_data, train_time_steps):
      X_train = create_X_data(train_data, train_time_steps)
      X_val = create_X_data(val_data, train_time_steps)
      X_test = create_X_data(test_data, train_time_steps)
      # reshape input to be [samples, time steps, features] which is required for LSTM
      X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
      X_val =X_val.reshape(X_val.shape[0],X_val.shape[1] , 1)
      X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

      X_train = torch.from_numpy(X_train).float()
      X_val = torch.from_numpy(X_val).float()
      X_test = torch.from_numpy(X_test).float()

      return X_train, X_val , X_test

def create_y_data(dataset, time_step=1):
    dataY = []
    for i in range(len(dataset)):
        y_data = np.max(dataset[i][time_step:])
        dataY.append(y_data)
    return np.array(dataY)

def ready_y_data(train_data, val_data, test_data, train_time_steps):
      y_train = create_y_data(train_data, train_time_steps)
      y_val = create_y_data(val_data, train_time_steps)
      y_test = create_y_data(test_data, train_time_steps )
      y_train =y_train.reshape(-1 , 1)
      y_val =y_val.reshape(-1 , 1)
      y_test = y_test.reshape(-1, 1)

      y_train = torch.from_numpy(y_train).float()
      y_val = torch.from_numpy(y_val).float()
      y_test = torch.from_numpy(y_test).float()

      return y_train , y_val, y_test

def inverse_scaler(predictions, actuals, scaler):
    predictions_inverse_scaler = scaler.inverse_transform(predictions)
    actuals_inverse_scaler = scaler.inverse_transform(actuals)
    return predictions_inverse_scaler, actuals_inverse_scaler

def main():
    args = parse_args()

    # Data Loading

    data_path = '../../datasets/'

    hurricanefile_nhc = data_path+'hurricane.mat'
    hurricane_data_nhc = scipy.io.loadmat(hurricanefile_nhc)
    hurricane_nhc = hurricane_data_nhc['hurricane']
    hurricanefile = data_path+'hurricane_1.mat'
    hurricane_data = scipy.io.loadmat(hurricanefile)
    hurricane = hurricane_data['hurricane']
    forecasts_data_file = data_path+'forecasts_int.mat'
    forecasts_data_mat = scipy.io.loadmat(forecasts_data_file)
    nhc_forecasts = forecasts_data_mat['NHC']
    time_split = forecasts_data_mat['time']
    model_forecasts = forecasts_data_mat['X']
    ground_truths = forecasts_data_mat['Y']
    best_track_file = data_path+'best_track.mat'
    best_track_matlab_data = scipy.io.loadmat(best_track_file)
    best_track = best_track_matlab_data['best_track']

    # Data Creation

    total_timesteps= 24
    train_time_steps = 16
    test_time_steps = total_timesteps - train_time_steps
    number_of_hurricanes = best_track[0].shape[0]

    nhc_hurricane_forecast_dict = {}
    nhc_original_dict = {}
    test_data_raw = []
    valid_nhc_forecasts = 0
    for i in range(time_split.shape[0]):
        nhc_hurricane_timesteps = time_split[i][1] - time_split[i][0] + 1
        if nhc_hurricane_timesteps >= total_timesteps:
            first_point_index = time_split[i][0] - 1
            prediction_window_start = first_point_index + train_time_steps - 1
            nhc_forecast = nhc_forecasts[0, 1:, prediction_window_start]
            hurricane_name = hurricane_nhc[0][i][0][0]
            if np.nansum(nhc_forecast) > 0:
                nhc_hurricane_forecast_dict[hurricane_name] = nhc_forecast
                nhc_original_dict[hurricane_name] = nhc_forecasts[0, 0,
                                                    prediction_window_start + 1:prediction_window_start + test_time_steps + 1]
                test_data_raw.append(nhc_forecasts[0, 0, first_point_index:first_point_index + total_timesteps])
                valid_nhc_forecasts += 1
            prediction_window_index = prediction_window_start + test_time_steps
            j = 1
            while prediction_window_index + test_time_steps < first_point_index + nhc_hurricane_timesteps:
                nhc_forecast = nhc_forecasts[0, 1:, prediction_window_index]
                if np.nansum(nhc_forecast) > 0:
                    key = hurricane_name + "_" + str(j + 1)
                    nhc_hurricane_forecast_dict[key] = nhc_forecast
                    nhc_original_dict[key] = nhc_forecasts[0, 0,
                                            prediction_window_index + 1:prediction_window_index + test_time_steps + 1]
                    test_data_raw.append(nhc_forecasts[0, 0,
                                        prediction_window_index + test_time_steps + 1 - total_timesteps:prediction_window_index + 1 + test_time_steps])
                    j += 1
                    valid_nhc_forecasts += 1
                prediction_window_index = prediction_window_index + test_time_steps

    total_observations = 0
    hurricane_count = 0
    for i in range(number_of_hurricanes):
        per_hurricane_observations = best_track[0][i].shape[0]
        if per_hurricane_observations>=total_timesteps:
            total_observations = total_observations + per_hurricane_observations - total_timesteps +1
            hurricane_count += 1
    print("hurricane_count, total_observations:", hurricane_count, total_observations)

    train_data = []
    test_data = []
    nhc_forecast_max = np.zeros(0)
    hurricane_original_best_track= {}
    nhc_count = 0
    hurricane_serial=0
    for i in range(number_of_hurricanes):
        per_hurricane_observations = best_track[0][i].shape[0]
        temp = []
        if per_hurricane_observations>=total_timesteps:
            for j in range(per_hurricane_observations):
                intensity = best_track[0][i][j][3]
                if j !=0:
                    if intensity  < 0 : intensity = best_track[0][i][j-1][3]
                temp.append(intensity)
    #             data[hurricane_serial,j]=intensity
            hurricane_serial+=1
        number_of_observations = len(temp)
        windows = 0
        neg_list = sum(n < 0 for n in temp)
        if neg_list>0: continue
        for k in range(0, number_of_observations+1-total_timesteps, test_time_steps):
            current_data = temp[k:k+total_timesteps]
            if k == 0:
                hurricane_key = hurricane[0][i][0][0]
            else: hurricane_key = hurricane[0][i][0][0]+"_"+str(windows+1)
            if hurricane_key in nhc_hurricane_forecast_dict:
                nhc_count  +=1
            else: train_data.append(current_data)
            windows +=1
    print("Number of Hurricanes:", hurricane_serial)
    print("Number of Hurricanes/Observations matched with NHC forecast:", nhc_count)
    print("After moving window, number of train data:", len(train_data))

    # threshold = np.percentile(np.array(train_data).flatten(), args.percentile, axis=0)

    # Train, Validate, Test Splits

    train_data = np.array(train_data)
    test_data = np.array(test_data_raw)
    print("Train and Test Data shape before normalizing/standardizing:", train_data.shape, test_data.shape)

    # scaler=MinMaxcaler(feature_range=(0,1))
    # train_data=scaler.fit_transform(train_data.reshape(-1,1))
    #
    scaler=StandardScaler()
    train_data=scaler.fit_transform(train_data.reshape(-1,1))

    train_data = train_data.reshape(-1,total_timesteps)
    print("Train Data shape after normalizing/standardizing:", train_data.shape)

    test_data=scaler.transform(test_data.reshape(-1,1))

    test_data = test_data.reshape(-1,total_timesteps)
    print("Train and Test Data shape before normalizing/standardizing:", train_data.shape, test_data.shape)

    print("Before Validation Data: train vs test", train_data.shape, test_data.shape)

    length = int(len(train_data)*0.8)
    random.shuffle(train_data)
    val_data= train_data[length:]
    train_data = train_data[0:length]
    print("After Validation Data (from train data): train vs validation vs test", train_data.shape, val_data.shape, test_data.shape)

    #Data Preprocessing
    X_train, X_val, X_test = ready_X_data(train_data, val_data, test_data, train_time_steps)

    y_train_max , y_val_max, y_test_max  = ready_y_data(train_data, val_data, test_data, train_time_steps)

    X_train_max, y_train_max = extend_last_batch(X_train, y_train_max, args.batch_size)
    X_val_max, y_val_max = extend_last_batch(X_val, y_val_max, args.batch_size)
    X_test_max, y_test_max = extend_last_batch(X_test, y_test_max, args.batch_size)

    X_train_full_max = torch.cat((X_train_max, X_val_max), 0)
    y_train_full_max = torch.cat((y_train_max, y_val_max), 0)

    # print(X_train_max.shape, y_train_max.shape, X_test_max.shape, y_test_max.shape)
    print('X_train_max:', X_train_max.shape, 'y_train_max:', y_train_max.shape)
    print('X_test_max:', X_test_max.shape, 'y_test_max:', y_test_max.shape)

    # visualize_distribution(y_train_max, y_test_max)

    # Calculate threshold
    threshold = np.percentile(X_train_full_max.flatten(), args.percentile, axis=0)
    threshold = torch.tensor(threshold, dtype=X_train_full_max.dtype, device=X_train_full_max.device)

    # threshold = scaler.transform([[threshold]])

    print("Threshold:", threshold)

    # DataLoader
    train_dataset = TensorDataset(X_train_max, y_train_max)
    test_dataset = TensorDataset(X_test_max, y_test_max)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    # Training configuration
    input_dim = 1 # Assuming 1 feature for the input data

    # Instantiate the model and optimizer
    vae = VAE(input_dim=input_dim, lstm_output_dim=args.output_dim, gcn_output_dim=0, latent_dim=args.latent_dim, future_steps=args.predict_steps, use_d_knn=args.use_d_knn, use_gcn=args.use_gcn, use_gpd=args.use_gpd, use_bernoulli=args.use_bernoulli)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    # Evaluate with batches
    all_truths = []
    all_predictions = []

    for epoch in range(args.epochs):
        vae.train()
        epoch_loss = 0
        total_samples = 0
        for X_batch, y_batch in train_loader:
            # print(f"X_batch shape: {X_batch.shape}, y_batch shape: {y_batch.shape}")
            if (y_batch == 0).sum().item() > 0:
                print(f"Số phần tử y bằng 0: {(y_batch == 0).sum().item()}")
            optimizer.zero_grad()
            forecasting, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero = vae(X_batch, X_batch, None)
            # print(y_batch.shape)
            # print(f"y_batch shape: {y_batch.shape}, forecasting shape: {forecasting.shape}")
            loss = vae.loss_function(forecasting, y_batch, z_mean_normal, z_log_var_normal, threshold, z_scale_extreme, z_shape_extreme, z_logits_zero)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)  # Scale by batch size
            total_samples += X_batch.size(0)  # Track total number of samples
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss / total_samples:.4f}")


    for X_batch, y_batch in test_loader:
        predictions, _, _, _, _, _ = vae(X_batch, X_batch, None)
        all_truths.append(y_batch)
        all_predictions.append(predictions)

    # Concatenate all predictions and truths
    all_truths = torch.cat(all_truths, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    print(f"All truths shape: {all_truths.shape}, All predictions shape: {all_predictions.shape}")

    # Calculate RMSE over the entire dataset
    average_rmse = rmse(all_predictions, all_truths)

    # Write all predictions vs truth to CSV
    with open('../../visualization/predictions_vs_truth.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Truth", "Predictions"])
        for truth, prediction in zip(all_truths, all_predictions):
            writer.writerow([truth.item(), prediction.item()])

    print(f"Test RMSE standardized: {average_rmse:.4f}")
    y_all, yhat_all = inverse_scaler(all_truths.tolist(), all_predictions.tolist(), scaler)
    print("RMSE of y : ", math.sqrt(mean_squared_error(y_all,yhat_all)))

if __name__ == "__main__":
    main()