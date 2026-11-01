"""
predict.py
----------
Interactive prediction script for both Stage 1 and Stage 2 models:
- Linear Regression (Stage 1)
- Neural Network with ReLU (Stage 2)

Allows you to enter your study habits and view predictions from one or both models.
"""

import os
import argparse
import numpy as np
from src.neural_network import NeuralNetwork


def load_linear_model(checkpoint_path: str = "model_checkpoint.npz"):
    """Loads trained weights, bias, and scaler for Linear Regression."""
    if not os.path.exists(checkpoint_path):
        return None
    data = np.load(checkpoint_path)
    return {
        "weights": data["weights"],
        "bias": float(data["bias"]),
        "scaler_mean": data["scaler_mean"],
        "scaler_std": data["scaler_std"]
    }


def load_neural_network_model(checkpoint_path: str = "nn_checkpoint.npz"):
    """Loads trained weights, biases, and scaler for Neural Network."""
    if not os.path.exists(checkpoint_path):
        return None
    nn = NeuralNetwork(input_dim=2, hidden_dim=8, output_dim=1)
    scaler_mean, scaler_std = nn.load(checkpoint_path)
    return {
        "model": nn,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std
    }


def predict_linear(study_hours: float, practice_questions: float, lr_bundle) -> float:
    raw = np.array([[study_hours, practice_questions]], dtype=float)
    scaled = (raw - lr_bundle["scaler_mean"]) / lr_bundle["scaler_std"]
    pred = np.dot(scaled, lr_bundle["weights"]) + lr_bundle["bias"]
    return float(np.clip(pred[0, 0], 0.0, 100.0))


def predict_neural_network(study_hours: float, practice_questions: float, nn_bundle) -> float:
    raw = np.array([[study_hours, practice_questions]], dtype=float)
    scaled = (raw - nn_bundle["scaler_mean"]) / nn_bundle["scaler_std"]
    pred = nn_bundle["model"].predict(scaled)
    return float(np.clip(pred[0, 0], 0.0, 100.0))


def interactive_mode(lr_bundle, nn_bundle):
    print("=" * 65)
    print("      STUDENT EXAM SCORE PREDICTOR (STAGE 1 & STAGE 2)")
    print("=" * 65)
    print("Available Models:")
    if lr_bundle:
        print("  [x] Linear Regression (model_checkpoint.npz)")
    else:
        print("  [ ] Linear Regression (not trained yet - run 'python train.py')")

    if nn_bundle:
        print("  [x] Neural Network   (nn_checkpoint.npz)")
    else:
        print("  [ ] Neural Network   (not trained yet - run 'python train_nn.py')")

    print("\nEnter your study habits below (or type 'q' or 'exit' to quit).\n")

    while True:
        try:
            hours_str = input("Enter study hours per week (e.g., 6.0): ").strip()
            if hours_str.lower() in ("q", "quit", "exit"):
                print("Goodbye!")
                break

            questions_str = input("Enter number of practice questions (e.g., 50): ").strip()
            if questions_str.lower() in ("q", "quit", "exit"):
                print("Goodbye!")
                break

            study_hours = float(hours_str)
            practice_questions = float(questions_str)

            if study_hours < 0 or practice_questions < 0:
                print(">> Please enter non-negative values.\n")
                continue

            print("\n" + "-" * 55)
            print(f"Inputs: {study_hours:.1f} hrs/week | {practice_questions:.0f} practice questions")
            if lr_bundle:
                score_lr = predict_linear(study_hours, practice_questions, lr_bundle)
                print(f">> [Linear Regression] Predicted Score: {score_lr:.2f} / 100")
            if nn_bundle:
                score_nn = predict_neural_network(study_hours, practice_questions, nn_bundle)
                print(f">> [Neural Network]    Predicted Score: {score_nn:.2f} / 100")
            print("-" * 55 + "\n")

        except ValueError:
            print(">> Invalid input! Please enter numeric values.\n")
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(description="Predict exam scores using Linear Regression or Neural Network.")
    parser.add_argument("--hours", type=float, help="Study hours per week")
    parser.add_argument("--questions", type=float, help="Practice questions completed")
    parser.add_argument("--model", type=str, choices=["both", "linear", "nn"], default="both",
                        help="Which model to use (default: both)")
    args = parser.parse_args()

    lr_bundle = load_linear_model()
    nn_bundle = load_neural_network_model()

    if lr_bundle is None and nn_bundle is None:
        print("Error: Neither model is trained yet! Run 'python train.py' and/or 'python train_nn.py'.")
        return

    # Direct CLI flag mode
    if args.hours is not None and args.questions is not None:
        print(f"Inputs: {args.hours:.1f} hrs/week | {args.questions:.0f} practice questions")
        if (args.model in ("both", "linear")) and lr_bundle:
            score_lr = predict_linear(args.hours, args.questions, lr_bundle)
            print(f"Linear Regression: {score_lr:.2f} / 100")
        if (args.model in ("both", "nn")) and nn_bundle:
            score_nn = predict_neural_network(args.hours, args.questions, nn_bundle)
            print(f"Neural Network:    {score_nn:.2f} / 100")
    else:
        interactive_mode(lr_bundle, nn_bundle)


if __name__ == "__main__":
    main()
