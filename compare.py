"""
compare.py
----------
Evaluates BOTH models on the EXACT same test data side-by-side:
1. Linear Regression (Stage 1)
2. Neural Network (Stage 2)

Allows you to directly see how the two architectures compare in accuracy and predictions.
"""

import os
import numpy as np
from src.dataset import load_dataset, train_test_split, StandardScaler
from src.model import LinearRegression
from src.neural_network import NeuralNetwork
from src.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score
)


def main():
    print("=" * 70)
    print("      MODEL COMPARISON: LINEAR REGRESSION VS NEURAL NETWORK")
    print("=" * 70)

    linear_checkpoint = "model_checkpoint.npz"
    nn_checkpoint = "nn_checkpoint.npz"

    if not os.path.exists(linear_checkpoint):
        print(f"Error: '{linear_checkpoint}' not found! Run 'python train.py' first.")
        return
    if not os.path.exists(nn_checkpoint):
        print(f"Error: '{nn_checkpoint}' not found! Run 'python train_nn.py' first.")
        return

    # Load dataset & identical test split
    X, y = load_dataset("data/student_scores.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=42)

    # 1. Load Linear Regression Model
    data_lr = np.load(linear_checkpoint)
    scaler_mean_lr = data_lr["scaler_mean"]
    scaler_std_lr = data_lr["scaler_std"]
    X_test_lr = (X_test - scaler_mean_lr) / scaler_std_lr
    y_pred_lr = np.dot(X_test_lr, data_lr["weights"]) + float(data_lr["bias"])
    y_pred_lr = np.clip(y_pred_lr, 0.0, 100.0)

    # 2. Load Neural Network Model
    nn = NeuralNetwork(input_dim=2, hidden_dim=8, output_dim=1)
    scaler_mean_nn, scaler_std_nn = nn.load(nn_checkpoint)
    X_test_nn = (X_test - scaler_mean_nn) / scaler_std_nn
    y_pred_nn = nn.predict(X_test_nn)
    y_pred_nn = np.clip(y_pred_nn, 0.0, 100.0)

    # 3. Calculate Metrics for both
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = root_mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    rmse_nn = root_mean_squared_error(y_test, y_pred_nn)
    r2_nn = r2_score(y_test, y_pred_nn)

    print("\nMETRICS COMPARISON (On 30 Unseen Test Students):")
    print("-" * 65)
    print(f"{'Metric':<28} | {'Linear Regression':<16} | {'Neural Network':<16}")
    print("-" * 65)
    print(f"{'Mean Absolute Error (MAE)':<28} | {mae_lr:14.4f} pts | {mae_nn:14.4f} pts")
    print(f"{'Root Mean Squared Error (RMSE)':<28} | {rmse_lr:14.4f} pts | {rmse_nn:14.4f} pts")
    print(f"{'R-squared (R^2) Accuracy':<28} | {r2_lr * 100:13.2f} %  | {r2_nn * 100:13.2f} %")
    print("-" * 65)

    print("\nSAMPLE PREDICTIONS COMPARISON:")
    print("-" * 75)
    print(f"{'Hours':>5} | {'Questions':>9} | {'Actual':>7} | {'Linear Pred':>11} | {'NN Pred':>11} | {'Winner':>8}")
    print("-" * 75)
    for i in range(min(8, len(y_test))):
        hours = X_test[i, 0]
        q = X_test[i, 1]
        act = y_test[i, 0]
        p_lr = y_pred_lr[i, 0]
        p_nn = y_pred_nn[i, 0]

        err_lr = abs(p_lr - act)
        err_nn = abs(p_nn - act)
        winner = "NN" if err_nn < err_lr else ("Tie" if abs(err_nn - err_lr) < 0.1 else "Linear")

        print(f"{hours:5.1f} | {q:9.0f} | {act:7.2f} | {p_lr:11.2f} | {p_nn:11.2f} | {winner:>8}")
    print("-" * 75)


if __name__ == "__main__":
    main()

