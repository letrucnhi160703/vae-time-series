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

def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE model with different hyperparameters.")
    
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

    return parser.parse_args()

def parse_hurricane_data(file_path, min_length=24):
    hurricanes = []
    hurricane_data = []
    header = None

    with open(file_path, 'r') as file:
        for line in file:
            if 'AL' in line:  # Header line for a new hurricane
                if header and len(hurricane_data) >= min_length:
                    wind_speeds = [float(row[6]) for row in hurricane_data if row[6] != '-999']
                    if len(wind_speeds) >= min_length:
                        hurricanes.append(wind_speeds)
                header = line
                hurricane_data = []
            else:
                hurricane_data.append(line.strip().split(','))

        # Process the last hurricane in the file
        if header and len(hurricane_data) >= min_length:
            wind_speeds = [float(row[6]) for row in hurricane_data if row[6] != '-999']
            if len(wind_speeds) >= min_length:
                hurricanes.append(wind_speeds)

    return hurricanes

def create_sliding_windows_from_np_array(data, window_length=12, predict_steps=1):
    X_data, Y_data = [], []
    for i in range(len(data) - window_length - predict_steps + 1):
        X_data.append(data[i:i+window_length])  # (window_length, num_features)
        # Y_data.append(data[i+window_length:i+window_length+predict_steps])  # (predict_steps, num_features)
        Y_data.append(data[i+window_length:i+window_length+predict_steps, 0]) # (predict_steps, 1) just predict the first feature [0]
    return np.array(X_data, dtype=np.float32), np.array(Y_data, dtype=np.float32).reshape(-1)
    # return np.array(X_data, dtype=np.float32), np.array(Y_data, dtype=np.float32).reshape(-1, predict_steps, 1) # just predict the first feature [0]

def create_sliding_windows_from_list(hurricanes, window_length=4, predict_steps=1):
    X_data, Y_data = [], []
    for hurricane in hurricanes:
        for i in range(len(hurricane) - window_length - predict_steps + 1):
            # Input: window_length steps
            X_data.append(hurricane[i:i+window_length])
            # Output: predict_steps steps
            Y_data.append(hurricane[i+window_length:i+window_length+predict_steps])
    return X_data, Y_data

def load_csv_dataset(file_path):
    df = pd.read_csv(file_path, delimiter=",")
    df = df.select_dtypes(include=[np.number])
    return df.values.astype(np.float32), df.columns

def rmse(predictions, targets):
    return torch.sqrt(F.mse_loss(predictions, targets))

def main():
    args = parse_args()
    # print(args)

    # file_path = '../../datasets/sunspots.csv'
    # file_path = '../../datasets/LD2011_2014_less.csv'
    if 'hurdat' in args.file_path:
        data = parse_hurricane_data(args.file_path)
        X_data, Y_data = create_sliding_windows_from_list(data, args.window_length, args.predict_steps)
        X_data = np.array(X_data, dtype=np.float32)
        Y_data = np.array(Y_data, dtype=np.float32)
    else:
        data, _ = load_csv_dataset(args.file_path)
            # Create Sliding Windows
        X_data, Y_data = create_sliding_windows_from_np_array(data, args.window_length, args.predict_steps)

    # # Sliding window parameters
    # window_length = 4
    # predict_steps = 1
    # percentile = 90

    # print(X_data.shape, Y_data.shape)
    threshold = np.percentile(Y_data, args.percentile, axis=0)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

    # Standardizing Data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    if 'hurdat' in args.file_path:
        num_features = 1
        X_train = scaler_X.fit_transform(X_train.reshape(-1, args.window_length)).reshape(-1, args.window_length, 1)
        y_train = scaler_y.fit_transform(y_train)

        X_test = scaler_X.transform(X_test.reshape(-1, args.window_length)).reshape(-1, args.window_length, 1)
        y_test = scaler_y.transform(y_test)

        threshold_scaled = scaler_y.transform(np.array([threshold]).reshape(-1, 1))
        threshold = threshold_scaled[0, 0]
        print(f"Threshold at Q {args.percentile}% is: {threshold}")
    else:
        num_features = X_train.shape[2]

        X_train = scaler_X.fit_transform(X_train.reshape(-1, num_features)).reshape(-1, args.window_length, num_features)
        # y_train = scaler_y.fit_transform(y_train.reshape(-1, num_features)).reshape(-1, predict_steps, num_features) # predict all features
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1, args.predict_steps) # just predict the first feature

        X_test = scaler_X.transform(X_test.reshape(-1, num_features)).reshape(-1, args.window_length, num_features)
        # y_test = scaler_y.transform(y_test.reshape(-1, num_features)).reshape(-1, predict_steps, num_features) # predict all features
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1, args.predict_steps) # just predict the first feature

        # print(y_train.shape, y_test.shape)

        # Scale threshold
        threshold_scaled = scaler_y.transform(np.array([[threshold]]))
        threshold = threshold_scaled[0, 0]
        print(f"Threshold at Q {args.percentile}% is: {threshold}")

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # DataLoader
    # batch_size = 16
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Training configuration
    input_dim = num_features
    # output_dim = 64
    # latent_dim = 32
    # epochs = 20
    # learning_rate = 1e-4

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
            if (y_batch == 0).sum().item() > 0:
                print(f"Số phần tử y bằng 0: {(y_batch == 0).sum().item()}")
            optimizer.zero_grad()
            forecasting, z_mean_normal, z_log_var_normal, z_scale_extreme, z_shape_extreme, z_logits_zero = vae(X_batch, X_batch, None)
            # print(y_batch.shape)
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

    # Calculate RMSE over the entire dataset
    average_rmse = rmse(all_predictions, all_truths).item()

    # Write all predictions vs truth to CSV
    with open('../../visualization/predictions_vs_truth.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Truth", "Predictions"])
        for truth, prediction in zip(all_truths, all_predictions):
            writer.writerow([truth.item(), prediction.item()])

    print(f"Test RMSE: {average_rmse:.4f}")

if __name__ == "__main__":
    main()