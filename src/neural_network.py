"""
neural_network.py
-----------------
A multi-layer neural network (Multi-Layer Perceptron / MLP) built completely
from scratch using NumPy.

Why upgrade from Linear Regression to a Neural Network?
1. Linear Regression has only 1 layer: y = X * W + b.
   It can only draw a straight line or flat plane.
2. A Neural Network adds a "Hidden Layer" between the inputs and output:
   Input (2 features) -> Hidden Layer (8 neurons + ReLU) -> Output (1 score)
3. Non-linear activation (ReLU):
   Enables the network to learn non-linear patterns, bends, and curves.
4. Backpropagation (Chain Rule):
   Allows the error from the output layer to travel backward to tune all hidden weights!
"""

import numpy as np


class NeuralNetwork:
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, output_dim: int = 1, learning_rate: float = 0.005):
        """
        Initializes the neural network architecture.
        
        Args:
            input_dim (int): Number of input features (2: study hours, questions).
            hidden_dim (int): Number of hidden neurons (e.g., 8).
            output_dim (int): Number of output targets (1: exam score).
            learning_rate (float): Step size for gradient descent updates.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Weight Initialization (He / Kaiming Initialization):
        # We multiply by sqrt(2 / fan_in) so values don't explode or vanish with ReLU.
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

        # Cache to store intermediate values during the forward pass
        # Needed during backpropagation to calculate gradients
        self.cache = {}
        self.loss_history = []

    def relu(self, Z: np.ndarray) -> np.ndarray:
        """
        Rectified Linear Unit (ReLU) Activation:
        Formula:
            ReLU(z) = max(0, z)
            
        Intuition: If a signal is negative, turn it off (0). If positive, let it pass.
        This simple switch is what gives neural networks the superpower of non-linearity!
        """
        return np.maximum(0, Z)

    def relu_derivative(self, Z: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU:
        Formula:
            d/dz ReLU(z) = 1 if z > 0 else 0
        """
        return (Z > 0).astype(float)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward Propagation:
        Information flows from inputs to hidden layer to output layer.
        
        Step 1 (Hidden Layer):
            Z1 = X @ W1 + b1          (Linear combination)
            A1 = ReLU(Z1)             (Non-linear activation)
            
        Step 2 (Output Layer):
            Z2 = A1 @ W2 + b2         (Linear combination to produce score)
            y_pred = Z2
        """
        # Layer 1
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)

        # Layer 2 (Output is continuous exam score, so no final activation needed)
        Z2 = np.dot(A1, self.W2) + self.b2
        y_pred = Z2

        # Save values for the backward pass
        self.cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2
        }
        return y_pred

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Backpropagation (The Chain Rule):
        Calculates gradients of the loss with respect to all weights and biases.
        
        Error flows backwards: Output Layer -> Hidden Layer -> Input
        """
        num_samples = y_true.shape[0]
        X = self.cache["X"]
        Z1 = self.cache["Z1"]
        A1 = self.cache["A1"]

        # 1. Output Layer Gradients (Layer 2)
        # Derivative of MSE Loss w.r.t Z2: dL/dZ2 = (2 / N) * (y_pred - y_true)
        dZ2 = (2.0 / num_samples) * (y_pred - y_true)

        # Gradients for Layer 2 parameters:
        # dW2 = A1.T @ dZ2
        # db2 = sum(dZ2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # 2. Hidden Layer Gradients (Layer 1)
        # Propagate error back through W2 into hidden layer: dA1 = dZ2 @ W2.T
        dA1 = np.dot(dZ2, self.W2.T)

        # Pass through ReLU derivative: dZ1 = dA1 * relu'(Z1)
        dZ1 = dA1 * self.relu_derivative(Z1)

        # Gradients for Layer 1 parameters:
        # dW1 = X.T @ dZ1
        # db1 = sum(dZ1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # 3. Update all parameters with Gradient Descent
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, verbose: bool = True):
        """
        Trains the neural network using forward passes and backpropagation.
        """
        num_samples = X.shape[0]
        self.loss_history = []

        if verbose:
            print(f"Training Neural Network ({self.input_dim} -> {self.hidden_dim} [ReLU] -> {self.output_dim})...")
            print(f"Total samples: {num_samples}, Epochs: {epochs}, Learning Rate: {self.learning_rate}\n")

        for epoch in range(1, epochs + 1):
            # 1. Forward Pass
            y_pred = self.forward(X)

            # 2. Compute Loss (MSE)
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)

            # 3. Backward Pass & Parameter Update
            self.backward(y, y_pred)

            # Print progress every 100 epochs
            if verbose and (epoch == 1 or epoch % 100 == 0 or epoch == epochs):
                print(f"Epoch [{epoch:4d}/{epochs}] - Loss (MSE): {loss:8.4f}")

        if verbose:
            print("\nNeural Network training completed successfully!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions on new inputs."""
        return self.forward(X)

    def save(self, filepath: str, scaler_mean: np.ndarray = None, scaler_std: np.ndarray = None):
        """Saves weights, biases, and optional scaler stats to an .npz file."""
        save_dict = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2
        }
        if scaler_mean is not None and scaler_std is not None:
            save_dict["scaler_mean"] = scaler_mean
            save_dict["scaler_std"] = scaler_std
        np.savez(filepath, **save_dict)
        print(f"Neural Network checkpoint saved to: {filepath}")

    def load(self, filepath: str):
        """Loads weights and biases from an .npz file."""
        data = np.load(filepath)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        scaler_mean = data["scaler_mean"] if "scaler_mean" in data else None
        scaler_std = data["scaler_std"] if "scaler_std" in data else None
        print(f"Neural Network checkpoint loaded from: {filepath}")
        return scaler_mean, scaler_std
