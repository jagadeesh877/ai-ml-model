"""
model.py
--------
This module defines the LinearRegression AI model from scratch using NumPy.

Core Concepts:
1. Weights (W): How much influence each feature has on the prediction.
   - For example: if study hours has a high weight, each extra hour adds many points.
2. Bias (b): The baseline starting value if all inputs were zero.
   - For example: the base exam score if a student studied 0 hours.
3. Prediction Formula:
   y_pred = (X @ W) + b
4. Loss (Mean Squared Error):
   Measures how far off our predictions are from the true answers.
5. Gradient Descent:
   Calculates the slope (gradients) and updates weights and bias to reduce the loss.
"""

import numpy as np


class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        Initializes the model.
        
        Args:
            learning_rate (float): Step size for gradient descent (how big each weight adjustment is).
            n_iterations (int): How many times the model cycles through the entire dataset (epochs).
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None  # Shape will be (num_features, 1)
        self.bias = None     # A single float number
        self.loss_history = []  # Tracks the loss at each iteration so we can see it decrease

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward Pass (Prediction step).
        
        Formula:
            y_pred = X @ W + b
            
        In NumPy:
            - np.dot(X, self.weights) or X @ self.weights multiplies each feature by its weight.
            - Adding self.bias adds the baseline offset to every prediction.
        """
        return np.dot(X, self.weights) + self.bias

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        Trains the model on training data (X, y) using Gradient Descent.
        
        Args:
            X (np.ndarray): Input features, shape (num_samples, num_features)
            y (np.ndarray): True target values, shape (num_samples, 1)
            verbose (bool): Whether to print progress every 100 iterations.
        """
        num_samples, num_features = X.shape

        # Step 1: Initialize weights to zeros (or small random numbers) and bias to 0.0
        self.weights = np.zeros((num_features, 1))
        self.bias = 0.0
        self.loss_history = []

        if verbose:
            print(f"Starting training for {self.n_iterations} iterations...")
            print(f"Number of training samples: {num_samples}, Features: {num_features}")
            print(f"Initial weights: {self.weights.flatten()}, Initial bias: {self.bias:.2f}\n")

        # Step 2: Training Loop (Iterative learning)
        for i in range(1, self.n_iterations + 1):
            # 2a. Make a guess (Forward Pass)
            y_pred = self.forward(X)

            # 2b. Compute how wrong the guess is (Loss calculation: Mean Squared Error)
            # error = y_pred - y
            error = y_pred - y
            current_loss = np.mean(error ** 2)
            self.loss_history.append(current_loss)

            # 2c. Compute Gradients (Calculus derivatives)
            # The gradient tells us the direction of the error:
            # - If gradient is positive, weight is too high -> decrease it.
            # - If gradient is negative, weight is too low -> increase it.
            # dW = (2 / N) * X.T @ error
            # db = (2 / N) * sum(error)
            dW = (2.0 / num_samples) * np.dot(X.T, error)
            db = (2.0 / num_samples) * np.sum(error)

            # 2d. Update weights and bias (Gradient Descent step)
            # Subtract gradient * learning_rate to move downhill toward lower error
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db

            # Print progress every 100 iterations (or on first and last)
            if verbose and (i == 1 or i % 100 == 0 or i == self.n_iterations):
                print(f"Epoch [{i:4d}/{self.n_iterations}] - Loss (MSE): {current_loss:8.4f} | "
                      f"Weights: [{self.weights[0, 0]:.4f}, {self.weights[1, 0]:.4f}] | "
                      f"Bias: {self.bias:7.4f}")

        if verbose:
            print("\nTraining completed successfully!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Uses the trained model weights and bias to make predictions on new data.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model is not trained yet! Please call model.fit() first.")
        return self.forward(X)

    def save(self, filepath: str):
        """Saves learned weights and bias to an .npz file."""
        np.savez(filepath, weights=self.weights, bias=self.bias)
        print(f"Model parameters saved to: {filepath}")

    def load(self, filepath: str):
        """Loads learned weights and bias from an .npz file."""
        data = np.load(filepath)
        self.weights = data["weights"]
        self.bias = float(data["bias"])
        print(f"Model parameters loaded from: {filepath}")

