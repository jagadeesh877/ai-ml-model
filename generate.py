"""
generate.py
-----------
Interactive text generation script for Stage 4: Bigram Language Model.

Allows you to enter custom prompt text, adjust the temperature (creativity dial),
and watch the model generate text character-by-character!
"""

import os
import argparse
from src.tokenizer import CharTokenizer
from src.bigram_model import BigramLanguageModel


def load_model_and_tokenizer(
    vocab_path: str = "bigram_vocab.json",
    checkpoint_path: str = "bigram_checkpoint.npz"
):
    if not os.path.exists(vocab_path):
        vocab_path = "tokenizer_vocab.json"
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file '{vocab_path}' not found! Run demo_tokenizer.py first.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found! Run train_bigram.py first.")

    tokenizer = CharTokenizer()
    tokenizer.load(vocab_path)

    model = BigramLanguageModel(vocab_size=tokenizer.vocab_size)
    model.load(checkpoint_path)

    return model, tokenizer


def interactive_mode(model: BigramLanguageModel, tokenizer: CharTokenizer):
    print("=" * 65)
    print("      STAGE 4: TEXT GENERATOR (BIGRAM NEURAL MODEL)")
    print("=" * 65)
    print("Enter a starting letter or word to prompt the model.")
    print("Type 'q' or 'exit' anytime to quit.\n")

    while True:
        try:
            prompt = input("Enter prompt (e.g., 'The', 'How', 'E'): ").strip()
            if prompt.lower() in ("q", "quit", "exit"):
                print("Goodbye!")
                break
            if not prompt:
                prompt = "T"

            length_str = input("How many characters to generate? (default 80): ").strip()
            length = int(length_str) if length_str else 80

            temp_str = input("Temperature [0.2 = focused, 0.8 = balanced, 1.2 = wild] (default 0.8): ").strip()
            temp = float(temp_str) if temp_str else 0.8

            print("\nGenerating...")
            print("-" * 55)
            generated = model.generate(tokenizer, start_char=prompt, max_new_tokens=length, temperature=temp)
            print(generated)
            print("-" * 55 + "\n")

        except ValueError:
            print(">> Please enter valid numbers for length and temperature.\n")
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(description="Generate text using the Stage 4 Bigram Language Model.")
    parser.add_argument("--prompt", type=str, default="The", help="Starting text prompt (default: 'The')")
    parser.add_argument("--length", type=int, default=100, help="Number of characters to generate (default: 100)")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()

    # If flags passed or run directly:
    if args.prompt != "The" or args.length != 100 or args.temp != 0.8:
        print(f"Prompt: '{args.prompt}' | Length: {args.length} | Temperature: {args.temp}")
        print("-" * 55)
        text = model.generate(tokenizer, start_char=args.prompt, max_new_tokens=args.length, temperature=args.temp)
        print(text)
        print("-" * 55)
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()

