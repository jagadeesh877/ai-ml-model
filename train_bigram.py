"""
train_bigram.py
---------------
Trains the Stage 4 Bigram Neural Language Model on data/tiny_story.txt.

This script:
1. Loads the text corpus and tokenizes it.
2. Creates bigram transition pairs (current_char -> next_char).
3. Evaluates baseline loss (untrained random guess = -log(1/V)).
4. Shows text generation BEFORE training (pure random noise).
5. Trains the model using Softmax and Cross-Entropy Loss with Gradient Descent.
6. Shows text generation AFTER training (learned common character combinations!).
7. Saves the model checkpoint to 'bigram_checkpoint.npz'.
"""

import os
import numpy as np
from src.tokenizer import CharTokenizer
from src.bigram_model import BigramLanguageModel


def main():
    print("=" * 70)
    print("      STAGE 4: TRAINING BIGRAM LANGUAGE MODEL (NUMPY)")
    print("=" * 70)

    corpus_path = os.path.join("data", "tiny_story.txt")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file '{corpus_path}' not found!")

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Fit Tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(text)
    encoded = np.array(tokenizer.encode(text), dtype=int)
    vocab_size = tokenizer.vocab_size

    print(f"Corpus length: {len(text)} characters")
    print(f"Vocabulary size: {vocab_size} unique characters")

    # 2. Prepare Bigram Dataset (current_char -> next_char)
    # X: all characters except the last
    # y: all characters except the first
    X = encoded[:-1]
    y = encoded[1:]
    num_transitions = len(X)
    print(f"Total bigram training pairs: {num_transitions}")

    # Theoretical baseline loss for random guessing:
    baseline_loss = -np.log(1.0 / vocab_size)
    print(f"Baseline loss (pure random guess = -log(1/{vocab_size})): {baseline_loss:.4f}\n")

    # 3. Initialize Model
    # A learning rate of 2.0 to 5.0 is standard for bigram frequency optimization
    model = BigramLanguageModel(vocab_size=vocab_size, learning_rate=3.0)

    # Initial generation before training (all weights zero -> uniform random)
    print("-" * 70)
    print("SAMPLE GENERATION BEFORE TRAINING (Random Guessing):")
    print("-" * 70)
    sample_untrained = model.generate(tokenizer, start_char="T", max_new_tokens=60, temperature=1.0, seed=42)
    print(repr(sample_untrained))
    print("-" * 70 + "\n")

    # 4. Train Model
    epochs = 400
    model.fit(X, y, epochs=epochs, verbose=True)

    # 5. Text Generation After Training
    print("\n" + "=" * 70)
    print("SAMPLE GENERATION AFTER TRAINING (Learned Language Statistics):")
    print("=" * 70)
    for temp in [0.5, 0.8, 1.2]:
        gen_text = model.generate(tokenizer, start_char="T", max_new_tokens=80, temperature=temp, seed=42)
        print(f"\n[Temperature = {temp}]")
        print(repr(gen_text))
    print("=" * 70 + "\n")

    # 6. Save Checkpoint
    checkpoint_file = "bigram_checkpoint.npz"
    model.save(checkpoint_file)
    print(f"Saved Bigram model weights to '{checkpoint_file}'")
    print("You can now run 'python generate.py' to generate text with your own custom prompts!")


if __name__ == "__main__":
    main()

