"""
train_transformer.py
--------------------
Stage 5: Trains a Tiny GPT-Style Transformer from Scratch on data/tiny_story.txt.

This script:
1. Loads the text corpus and tokenizes it using CharTokenizer.
2. Splits the corpus sequentially into 85% Train and 15% Validation (no future leakage).
3. Prepares sequence chunks of length (context_length + 1) where:
   - Input X: tokens[0 : T]
   - Target y: tokens[1 : T + 1] (every position predicts its next token!)
4. Initializes a TinyGPTLanguageModel (d_model=32, n_heads=2, d_ff=64, context_length=16).
5. Trains using AdamOptimizer with analytical backpropagation and gradient clipping.
6. Periodically evaluates on the validation split and logs train/val loss.
7. Demonstrates text generation before and after training.
8. Saves the model to 'transformer_checkpoint.npz'.
"""

import os
import numpy as np
from src.tokenizer import CharTokenizer
from src.transformer_model import TinyGPTLanguageModel, AdamOptimizer


def prepare_sequences(encoded_tokens: np.ndarray, context_length: int, step: int = 4):
    """
    Creates input sequences (X) and next-token targets (y).
    Each chunk has length (context_length + 1).
    X = chunk[:-1], y = chunk[1:]
    """
    X_list = []
    y_list = []
    chunk_len = context_length + 1

    for i in range(0, len(encoded_tokens) - chunk_len + 1, step):
        chunk = encoded_tokens[i : i + chunk_len]
        X_list.append(chunk[:-1])
        y_list.append(chunk[1:])

    if not X_list:
        raise ValueError("Encoded tokens shorter than context_length + 1!")

    return np.array(X_list, dtype=int), np.array(y_list, dtype=int)


def evaluate_loss(model: TinyGPTLanguageModel, X_val: np.ndarray, y_val: np.ndarray, batch_size: int = 16) -> float:
    """
    Computes average cross-entropy loss over validation sequences without updating weights.
    """
    num_samples = len(X_val)
    total_loss = 0.0
    num_batches = 0

    for i in range(0, num_samples, batch_size):
        x_batch = X_val[i : i + batch_size]
        y_batch = y_val[i : i + batch_size]
        _, probs, _ = model.forward(x_batch)
        loss = model.compute_loss(probs, y_batch)
        total_loss += loss * len(x_batch)
        num_batches += len(x_batch)

    return total_loss / max(num_batches, 1)


def main():
    print("=" * 70)
    print("      STAGE 5: TRAINING TINY GPT TRANSFORMER FROM SCRATCH")
    print("=" * 70)

    corpus_path = os.path.join("data", "tiny_story.txt")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file '{corpus_path}' not found!")

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(text)
    encoded = np.array(tokenizer.encode(text), dtype=int)
    vocab_size = tokenizer.vocab_size

    print(f"Corpus length: {len(text)} characters")
    print(f"Vocabulary size: {vocab_size} unique characters")

    # 2. Sequential Train / Validation Split (85% Train, 15% Val)
    # Strictly sequential: no random shuffling across time to prevent future data leakage!
    split_idx = int(len(encoded) * 0.85)
    train_tokens = encoded[:split_idx]
    val_tokens = encoded[split_idx:]

    print(f"Training split:   {len(train_tokens)} characters")
    print(f"Validation split: {len(val_tokens)} characters")

    # 3. Prepare Sequences
    context_length = 16
    X_train, y_train = prepare_sequences(train_tokens, context_length=context_length, step=2)
    X_val, y_val = prepare_sequences(val_tokens, context_length=context_length, step=2)

    print(f"Training batches:   {len(X_train)} sequences of length {context_length}")
    print(f"Validation batches: {len(X_val)} sequences of length {context_length}\n")

    # 4. Model Configuration
    d_model = 32
    n_heads = 2
    d_ff = 64
    epochs = 120
    learning_rate = 0.003
    batch_size = 16

    print(f"Model Architecture:")
    print(f"  - Context Window: {context_length} tokens")
    print(f"  - Embedding Dim (d_model): {d_model}")
    print(f"  - Attention Heads: {n_heads} (head dimension = {d_model // n_heads})")
    print(f"  - Feed-Forward Dim: {d_ff}")
    print(f"  - Number of Transformer Blocks: 1")
    print(f"  - Training Epochs: {epochs}")

    # Baseline theoretical loss (pure random guessing)
    baseline_loss = -np.log(1.0 / vocab_size)
    print(f"  - Theoretical Baseline Loss (-log(1/{vocab_size})): {baseline_loss:.4f}\n")

    # Initialize model
    model = TinyGPTLanguageModel(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff
    )

    optimizer = AdamOptimizer(learning_rate=learning_rate, grad_clip=1.0)

    # Sample generation before training
    print("-" * 70)
    print("SAMPLE GENERATION BEFORE TRAINING (Random Initialization):")
    print("-" * 70)
    sample_before = model.generate(tokenizer, prompt="The ", max_new_tokens=60, temperature=1.0, seed=42)
    print(repr(sample_before))
    print("-" * 70 + "\n")

    initial_train_loss = evaluate_loss(model, X_train, y_train, batch_size=batch_size)
    initial_val_loss = evaluate_loss(model, X_val, y_val, batch_size=batch_size)
    print(f"Initial Train Loss: {initial_train_loss:.4f} | Initial Val Loss: {initial_val_loss:.4f}\n")

    # 5. Training Loop
    num_samples = len(X_train)
    np.random.seed(42)

    for epoch in range(1, epochs + 1):
        # Shuffle batch order each epoch
        indices = np.random.permutation(num_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            x_batch = X_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]

            # Forward pass
            logits, probs, cache = model.forward(x_batch)

            # Compute loss
            loss = model.compute_loss(probs, y_batch)
            if not np.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}: {loss}")

            epoch_loss += loss * len(x_batch)
            num_batches += len(x_batch)

            # Backward pass (Analytical gradients)
            grads = model.backward(y_batch, cache)

            # Check gradients are finite
            for name, g in grads.items():
                if not np.all(np.isfinite(g)):
                    raise RuntimeError(f"Non-finite gradient in {name} at epoch {epoch}!")

            # Optimizer step
            param_map = model.get_params_and_grads(grads)
            optimizer.step(param_map)

        train_loss = epoch_loss / num_batches

        # Log progress periodically
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            val_loss = evaluate_loss(model, X_val, y_val, batch_size=batch_size)
            print(f"Epoch [{epoch:3d}/{epochs}] - Train Loss: {train_loss:7.4f} | Val Loss: {val_loss:7.4f}")

    final_train_loss = evaluate_loss(model, X_train, y_train, batch_size=batch_size)
    final_val_loss = evaluate_loss(model, X_val, y_val, batch_size=batch_size)

    print("\n" + "=" * 70)
    print("      TRAINING COMPLETE - SUMMARY RESULTS")
    print("=" * 70)
    print(f"Initial Train Loss: {initial_train_loss:.4f}  ->  Final Train Loss: {final_train_loss:.4f}")
    print(f"Initial Val Loss:   {initial_val_loss:.4f}  ->  Final Val Loss:   {final_val_loss:.4f}")
    print(f"Loss reduction:    {((initial_train_loss - final_train_loss) / initial_train_loss) * 100:.1f}%\n")

    # 6. Sample generation after training
    print("SAMPLE GENERATIONS AFTER TRAINING:")
    print("-" * 70)
    for temp in [0.5, 0.8, 1.0]:
        gen = model.generate(tokenizer, prompt="The ", max_new_tokens=80, temperature=temp, seed=42)
        print(f"[Temperature = {temp}]")
        print(repr(gen))
        print()
    print("-" * 70)

    # 7. Save Checkpoint Separately
    checkpoint_file = "transformer_checkpoint.npz"
    model.save_checkpoint(checkpoint_file)
    print(f"\nSaved Stage 5 checkpoint to '{checkpoint_file}'")
    print("Run 'python generate_transformer.py' to generate text from your trained Transformer!")


if __name__ == "__main__":
    main()

