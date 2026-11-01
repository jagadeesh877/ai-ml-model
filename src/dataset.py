"""
dataset.py
----------
This module handles all data operations:
1. Loading the dataset from a CSV file.
2. Splitting data into Training (80%) and Testing (20%) sets.
3. Feature scaling (Standardization: (x - mean) / std).
"""

import numpy as np


def load_dataset(csv_path: str):
    """
    Loads data from a CSV file using NumPy.
    
    Returns:
        X (np.ndarray): The input features (study_hours, practice_questions), shape (N, 2)
        y (np.ndarray): The target labels (exam_score), shape (N, 1)
    """
    # np.genfromtxt reads numeric tables from a text/csv file
    # skip_header=1 skips the column names row (study_hours,practice_questions,exam_score)
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)

    # Columns 0 and 1 are the features (X)
    X = data[:, 0:2]

    # Column 2 is the target we want to predict (y)
    # .reshape(-1, 1) ensures it is a 2D column vector with shape (N, 1) instead of 1D (N,)
    y = data[:, 2].reshape(-1, 1)

    return X, y


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 42):
    """
    Randomly splits the dataset into Training and Testing sets.
    
    Why split?
    Imagine studying for an exam:
    - Training data: The practice exercises you study with answers provided.
    - Testing data: The final exam questions you have never seen before!
    This checks if your model truly learned general patterns or just memorized.
    """
    np.random.seed(seed)
    
    # Total number of examples (rows)
    total_samples = X.shape[0]
    
    # Generate a shuffled list of indices: e.g., [42, 7, 103, 12, ...]
    shuffled_indices = np.random.permutation(total_samples)
    
    # Calculate how many rows should be in the test set
    test_size = int(total_samples * test_ratio)
    
    # Pick the test indices and train indices
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    
    # Slice the arrays using the chosen indices
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test


class StandardScaler:
    """
    Scales features so they have:
    - Mean (average) = 0
    - Standard Deviation (spread) = 1
    
    Why is this important?
    - Study hours ranges from 1 to 10.
    - Practice questions ranges from 10 to 100.
    Because practice questions is ~10x larger, the model might accidentally focus
    only on questions and ignore study hours. Scaling puts all features on equal footing.
    
    Formula:
        z = (x - mean) / std
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        """Calculates the mean and standard deviation for each column."""
        # axis=0 calculates across rows for each individual column
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        # Prevent division by zero if a feature has zero variance
        self.std = np.where(self.std == 0, 1.0, self.std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Applies the scaling formula using the previously calculated mean and std."""
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Combines fit and transform in one convenient call."""
        self.fit(X)
        return self.transform(X)

