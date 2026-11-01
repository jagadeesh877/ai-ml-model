"""
bigram_model.py
---------------
Stage 4: Bigram Neural Language Model & Text Generation (Pure NumPy).

What is a Bigram Language Model?
A model that predicts the probability of the NEXT character given ONLY the CURRENT character.
For example:
- Given 'T', what is the probability of 'h'? (High!)
- Given 'q', what is the probability of 'u'? (Very high!)
- Given ' ', what is the probability of 'z'? (Low!)

Why is this a crucial stepping stone to GPT?
1. GPT is an autoregressive next-token prediction model.
2. The classification head of GPT uses the exact same mathematics:
   - Logits (raw scores)
   - Softmax function (converting scores into probabilities)
   - Cross-Entropy Loss (scoring how well probabilities match the true target)
   - Temperature-scaled sampling (generating new creative text)
"""

import numpy as np
from typing import List, Tuple, Optional


class BigramLanguageModel:
    def __init__(self, vocab_size: int, learning_rate: float = 0.1):
        """
        Initializes the Bigram Neural Model.
        
        Args:
            vocab_size (int): Total number of unique characters in the vocabulary (V).
            learning_rate (float): Step size for gradient descent.
        """
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        # Weight matrix of shape (V, V):
        # Row i contains the score for every possible next character when current character is i.
        # Initialized to zeros (or small random values) so all transitions start equally likely.
        np.random.seed(42)
        self.W = np.zeros((vocab_size, vocab_size))
        self.b = np.zeros((1, vocab_size))

        self.loss_history = []

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """
        Converts raw scores (logits) into valid probabilities that sum to 1.0.
        
        Formula:
            P(k) = exp(z_k) / sum(exp(z_j))
            
        Numerical Stability Trick:
        Subtracting max(logits) prevents overflow (np.exp of a large number like 1000 explodes to inf).
        """
        # Subtract max along rows for numerical stability
        shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        probabilities = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        return probabilities

    def forward(self, x_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward Pass:
        Given an array of input character token IDs (shape: (N,)),
        computes the logits and next-character probabilities (shape: (N, V)).
        """
        # In a bigram model, input x is just the current token index.
        # W[x_indices] is equivalent to one_hot(x) @ W (fast row lookup!)
        logits = self.W[x_indices] + self.b
        probs = self.softmax(logits)
        return logits, probs

    def compute_loss(self, probs: np.ndarray, y_indices: np.ndarray) -> float:
        """
        Calculates Cross-Entropy Loss (Negative Log Likelihood).
        
        Formula:
            Loss = - (1 / N) * sum(log(P(true_next_char)))
            
        Intuition:
        If the model gave the true next character a probability of 1.0 (100%), loss is -log(1) = 0.
        If the model gave it a low probability like 0.01, loss is -log(0.01) = 4.6 (high penalty!).
        """
        num_samples = len(y_indices)
        # Extract the probability the model assigned to the actual true target
        # Add a tiny epsilon (1e-15) to prevent log(0)
        correct_probs = probs[np.arange(num_samples), y_indices]
        loss = -np.mean(np.log(correct_probs + 1e-15))
        return float(loss)

    def backward(self, probs: np.ndarray, x_indices: np.ndarray, y_indices: np.ndarray):
        """
        Backpropagation for Softmax + Cross-Entropy Loss:
        
        Derivative:
            dLogits = (1 / N) * (P - Y_one_hot)
            
        Remarkable fact: The gradient is simply:
        (Predicted Probabilities) minus (1 at the true character index)!
        """
        num_samples = len(y_indices)
        
        # dLogits starts as the predicted probabilities
        dLogits = probs.copy()
        
        # Subtract 1 from the correct character's probability
        dLogits[np.arange(num_samples), y_indices] -= 1.0
        dLogits /= num_samples

        # Gradient for bias: sum across samples
        db = np.sum(dLogits, axis=0, keepdims=True)

        # Gradient for weights:
        # Since x_indices are token IDs, we accumulate gradients into row x_indices
        dW = np.zeros_like(self.W)
        np.add.at(dW, x_indices, dLogits)

        # Gradient descent parameter update
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def fit(self, x_indices: np.ndarray, y_indices: np.ndarray, epochs: int = 500, verbose: bool = True):
        """
        Trains the bigram model on pairs of (current_token, next_token).
        """
        num_samples = len(x_indices)
        self.loss_history = []

        if verbose:
            print(f"Training Bigram Model on {num_samples} character transitions...")
            print(f"Vocabulary Size: {self.vocab_size}, Epochs: {epochs}, Learning Rate: {self.learning_rate}\n")

        for epoch in range(1, epochs + 1):
            # 1. Forward Pass
            logits, probs = self.forward(x_indices)

            # 2. Compute Cross-Entropy Loss
            loss = self.compute_loss(probs, y_indices)
            self.loss_history.append(loss)

            # 3. Backward Pass & Parameter Update
            self.backward(probs, x_indices, y_indices)

            # Print progress every 50 epochs
            if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == epochs):
                print(f"Epoch [{epoch:4d}/{epochs}] - Cross-Entropy Loss: {loss:7.4f}")

        if verbose:
            print("\nBigram training completed successfully!")

    def generate(
        self,
        tokenizer,
        start_char: str = "T",
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        seed: Optional[int] = None
    ) -> str:
        """
        Autoregressive Text Generation:
        Generates new text character-by-character using the trained transition weights.
        
        Args:
            tokenizer: CharTokenizer instance with encode/decode methods.
            start_char (str): Initial seed character or prompt.
            max_new_tokens (int): How many characters to generate.
            temperature (float):
                - temperature < 1.0: More conservative, deterministic, and repetitive.
                - temperature = 1.0: Standard probability distribution.
                - temperature > 1.0: More creative, varied, and random.
            seed (int): Optional random seed for reproducible generation.
        """
        if seed is not None:
            np.random.seed(seed)

        if not start_char:
            start_char = " "

        # Encode start character (use the last character of the prompt as current state)
        encoded_start = tokenizer.encode(start_char)
        current_token = encoded_start[-1]
        generated_ids = list(encoded_start)

        for _ in range(max_new_tokens):
            # 1. Forward pass for current character
            logits = self.W[current_token] + self.b[0]

            # 2. Apply Temperature: scale logits before softmax
            # Lower temperature sharpens the peak; higher temperature flattens it
            scaled_logits = logits / max(temperature, 1e-5)
            probs = self.softmax(scaled_logits)

            # 3. Sample next token from the probability distribution
            next_token = np.random.choice(self.vocab_size, p=probs)

            # 4. Append and advance
            generated_ids.append(next_token)
            current_token = next_token

        return tokenizer.decode(generated_ids)

    def save(self, filepath: str):
        """Saves weights and bias to an .npz file."""
        np.savez(filepath, W=self.W, b=self.b, vocab_size=self.vocab_size)
        print(f"Bigram model checkpoint saved to: {filepath}")

    def load(self, filepath: str):
        """Loads weights and bias from an .npz file."""
        data = np.load(filepath)
        self.W = data["W"]
        self.b = data["b"]
        self.vocab_size = int(data["vocab_size"])
        print(f"Bigram model checkpoint loaded from: {filepath}")

