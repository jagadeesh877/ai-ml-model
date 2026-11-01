"""
train.py
--------
This script runs the entire training pipeline:
1. Loads the student dataset.
2. Splits data into 80% Training and 20% Testing sets.
3. Scales features to have mean=0 and std=1.
4. Trains the Linear Regression model using Gradient Descent.
5. Evaluates the model on unseen test data.
6. Saves the trained weights and scaler parameters to disk.
"""

import os
import numpy as np
from src.dataset import load_dataset, train_test_split, StandardScaler
from src.model import LinearRegression
from src.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score
)


def main():
    print("=" * 60)
    print("      STEP 1: LOADING & PREPARING THE DATASET")
    print("=" * 60)

    dataset_path = os.path.join("data", "student_scores.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run generate_dataset.py first.")

    X, y = load_dataset(dataset_path)
    print(f"Total dataset shape: X = {X.shape} (inputs), y = {y.shape} (targets)")

    # Split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")

    # Feature Scaling: Standardize using training set statistics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Feature means: [Study Hours: {scaler.mean[0]:.2f}, Questions: {scaler.mean[1]:.2f}]")
    print(f"Feature stds:  [Study Hours: {scaler.std[0]:.2f}, Questions: {scaler.std[1]:.2f}]")

    print("\n" + "=" * 60)
    print("      STEP 2: TRAINING THE MODEL WITH GRADIENT DESCENT")
    print("=" * 60)

    # Initialize model with learning rate of 0.05 and 1000 iterations
    model = LinearRegression(learning_rate=0.05, n_iterations=1000)
    model.fit(X_train_scaled, y_train, verbose=True)

    print("\n" + "=" * 60)
    print("      STEP 3: EVALUATING ON UNSEEN TEST DATA")
    print("=" * 60)

    # Predict on test set
    y_test_pred = model.predict(X_test_scaled)

    # Compute metrics
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = root_mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    print(f"Mean Squared Error (MSE):       {mse:8.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:8.4f} points")
    print(f"Mean Absolute Error (MAE):      {mae:8.4f} points")
    print(f"R-squared (R^2) Accuracy Score: {r2 * 100:8.2f}%")

    print("\nSample Predictions on Unseen Test Data:")
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

    print("\n" + "=" * 60)
    print("      STEP 4: SAVING MODEL & SCALER PARAMETERS")
    print("=" * 60)

    checkpoint_path = "model_checkpoint.npz"
    np.savez(
        checkpoint_path,
        weights=model.weights,
        bias=model.bias,
        scaler_mean=scaler.mean,
        scaler_std=scaler.std
    )
    print(f"Saved trained model and scaler to '{checkpoint_path}'")
    print("You can now run 'python predict.py' to test interactive predictions!")


if __name__ == "__main__":
    main()
