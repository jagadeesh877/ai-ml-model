"""
train_nn.py
-----------
Trains the Stage 2 Neural Network on the student score dataset.
Saves weights and normalization statistics to 'nn_checkpoint.npz'
leaving the original Linear Regression checkpoint 'model_checkpoint.npz' untouched.
"""

import os
import numpy as np
from src.dataset import load_dataset, train_test_split, StandardScaler
from src.neural_network import NeuralNetwork
from src.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score
)


def main():
    print("=" * 65)
    print("      STAGE 2: TRAINING NEURAL NETWORK FROM SCRATCH")
    print("=" * 65)

    dataset_path = os.path.join("data", "student_scores.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # 1. Load Dataset
    X, y = load_dataset(dataset_path)

    # 2. Split (Same seed=42 as train.py to ensure identical test data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=42)

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Initialize Neural Network (2 inputs -> 8 hidden neurons with ReLU -> 1 output)
    # Using learning_rate = 0.005 and 1200 epochs for smooth convergence
    nn = NeuralNetwork(input_dim=2, hidden_dim=8, output_dim=1, learning_rate=0.005)

    # 5. Train with Backpropagation
    nn.fit(X_train_scaled, y_train, epochs=1200, verbose=True)

    print("\n" + "=" * 65)
    print("      NEURAL NETWORK EVALUATION ON UNSEEN TEST DATA")
    print("=" * 65)

    # 6. Predict on Test Data
    y_test_pred = nn.predict(X_test_scaled)
    y_test_pred = np.clip(y_test_pred, 0.0, 100.0)

    mse = mean_squared_error(y_test, y_test_pred)
    rmse = root_mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    print(f"Mean Squared Error (MSE):       {mse:8.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:8.4f} points")
    print(f"Mean Absolute Error (MAE):      {mae:8.4f} points")
    print(f"R-squared (R^2) Accuracy Score: {r2 * 100:8.2f}%")

    print("\nSample Test Predictions (Actual vs Neural Network):")
    print("-" * 55)
    print(f"{'Study Hours':>12} | {'Questions':>10} | {'Actual':>8} | {'Predicted':>10} | {'Error':>7}")
    print("-" * 55)
    for i in range(min(7, len(y_test))):
        hours = X_test[i, 0]
        questions = X_test[i, 1]
        actual = y_test[i, 0]
        pred = y_test_pred[i, 0]
        diff = pred - actual
        print(f"{hours:12.1f} | {questions:10.0f} | {actual:8.2f} | {pred:10.2f} | {diff:+7.2f}")
    print("-" * 55)

    # 7. Save Checkpoint Separately
    checkpoint_path = "nn_checkpoint.npz"
    nn.save(checkpoint_path, scaler_mean=scaler.mean, scaler_std=scaler.std)
    print(f"\nSaved neural network checkpoint to '{checkpoint_path}'")
    print("Run 'python compare.py' to compare Linear Regression vs Neural Network!")


if __name__ == "__main__":
    main()
