"""
metrics.py
----------
This module provides evaluation metrics implemented from scratch with NumPy.
These allow us to measure how accurate our AI model's predictions are.
"""

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates Mean Squared Error (MSE).
    
    Formula:
        MSE = (1 / N) * sum((y_true - y_pred) ** 2)
        
    Why square the errors?
    1. It eliminates negative signs (being 5 points too high or too low both count as an error).
    2. It heavily penalizes large mistakes (an error of 10 points produces 100 penalty points).
    """
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates Root Mean Squared Error (RMSE).
    
    Formula:
        RMSE = sqrt(MSE)
        
    Benefit: Unlike MSE, RMSE is in the exact same unit as our target (Exam Points).
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates Mean Absolute Error (MAE).
    
    Formula:
        MAE = (1 / N) * sum(|y_true - y_pred|)
        
    Intuition: The most human-readable metric!
    "On average, the model's guess is off by X points."
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the R-squared (R²) score, also known as Coefficient of Determination.
    
    Formula:
        R² = 1 - (Residual Sum of Squares / Total Sum of Squares)
        R² = 1 - (sum((y_true - y_pred) ** 2) / sum((y_true - mean(y_true)) ** 2))
        
    Intuition:
        - 1.0 (100%): Perfect model (makes zero errors).
        - 0.0: The model is no better than just guessing the average score every time.
        - Negative: The model is performing worse than guessing the average.
    """
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    total_sum_of_squares = np.sum((y_true - np.mean(y_true)) ** 2)

    # Avoid zero division if all target values happen to be identical
    if total_sum_of_squares == 0:
        return 1.0

    return float(1.0 - (residual_sum_of_squares / total_sum_of_squares))

