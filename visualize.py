import pandas as pd
import matplotlib.pyplot as plt

def load_results(filename='predictions_vs_truth.csv'):
    return pd.read_csv(filename)

########### All sample

def plot_predictions_vs_truth(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Truth'], label='Truth', color='b', linestyle='-')
    plt.plot(df['Predictions'], label='Predictions', color='r', linestyle='-')
    plt.title('Model Predictions vs Truth')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    results_df = load_results('predictions_vs_truth.csv')
    plot_predictions_vs_truth(results_df)


########### Some sample

def plot_partial_predictions_vs_truth(df, num_samples=200):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Truth'][:num_samples], label='Truth', color='b', linestyle='-')
    plt.plot(df['Predictions'][:num_samples], label='Predictions', color='r', linestyle='-')
    plt.title('Model Predictions vs Truth (Partial View)')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

results_df = load_results('predictions_vs_truth.csv')
plot_partial_predictions_vs_truth(results_df, num_samples=100)
