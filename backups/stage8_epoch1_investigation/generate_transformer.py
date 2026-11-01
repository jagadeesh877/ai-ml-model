"""
generate_transformer.py
-----------------------
Stage 7: Text Generation from Trained Tiny GPT Transformer Checkpoint.

Features:
- Seamless support for Byte-Pair Encoding (BPE) subword models and character models.
- Auto-detects tokenizer type (BPE vs Char) from vocabulary JSON.
- Checkpoint & vocabulary compatibility validation (guards against shape mismatches).
- Single prompt generation or temperature sweep (--temp-sweep).
- Repetition penalty to discourage repetitive loops.
- Top-K sampling option.
- Interactive mode and CLI flag modes.
- Clear error handling for unsupported characters and missing files.
"""

import os
import json
import argparse
from typing import Optional
from src.tokenizer import CharTokenizer, BPETokenizer
from src.transformer_model import TinyGPTLanguageModel


def load_transformer(
    checkpoint_path: Optional[str] = None,
    vocab_path: Optional[str] = None
):
    """Loads tokenizer and reconstructed transformer model with compatibility checks."""
    # Auto-resolve checkpoint path
    if checkpoint_path is None:
        if os.path.exists("transformer_bpe_checkpoint.npz"):
            checkpoint_path = "transformer_bpe_checkpoint.npz"
        elif os.path.exists("transformer_checkpoint.npz"):
            checkpoint_path = "transformer_checkpoint.npz"
        else:
            raise FileNotFoundError(
                "No checkpoint file found (neither 'transformer_bpe_checkpoint.npz' nor 'transformer_checkpoint.npz').\n"
                "Please run 'python train_transformer.py' first to train the model."
            )

    # Auto-resolve vocab path
    if vocab_path is None:
        if "bpe" in checkpoint_path and os.path.exists("bpe_vocab.json"):
            vocab_path = "bpe_vocab.json"
        elif os.path.exists("bpe_vocab.json"):
            vocab_path = "bpe_vocab.json"
        elif os.path.exists("transformer_vocab.json"):
            vocab_path = "transformer_vocab.json"
        elif os.path.exists("tokenizer_vocab.json"):
            vocab_path = "tokenizer_vocab.json"
        else:
            raise FileNotFoundError(
                "No vocabulary file found!\n"
                "Please run 'python train_transformer.py' first."
            )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found!")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file '{vocab_path}' not found!")

    # 1. Detect tokenizer type from vocabulary file
    with open(vocab_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if meta.get("tokenizer_type") == "bpe" or "merges" in meta:
        tokenizer = BPETokenizer()
        tokenizer.load(vocab_path)
    else:
        tokenizer = CharTokenizer()
        tokenizer.load(vocab_path)

    # 2. Reconstruct model and strictly validate checkpoint vocabulary size
    model = TinyGPTLanguageModel(vocab_size=tokenizer.vocab_size)
    model.load_checkpoint(checkpoint_path, expected_vocab_size=tokenizer.vocab_size)

    return model, tokenizer


def run_temperature_sweep(
    model: TinyGPTLanguageModel,
    tokenizer,
    prompt: str,
    length: int,
    rep_penalty: float = 1.1,
    seed: int = 42
):
    """Generates text across a range of sampling temperatures to compare coherence vs diversity."""
    temperatures = [0.3, 0.6, 0.8, 1.2]
    tok_type = "BPE Subword" if isinstance(tokenizer, BPETokenizer) else "Character"
    print("=" * 70)
    print(f"TEMPERATURE SWEEP ({tok_type} Model, Vocab={tokenizer.vocab_size})")
    print(f"Prompt: {repr(prompt)}")
    if isinstance(tokenizer, BPETokenizer):
        subwords = tokenizer.tokenize_to_strings(prompt)
        print(f"Prompt Subwords: {subwords}")
    print("=" * 70)

    for temp in temperatures:
        print(f"\n[Temperature = {temp:.1f}]")
        print("-" * 55)
        text = model.generate(
            tokenizer,
            prompt=prompt,
            max_new_tokens=length,
            temperature=temp,
            repetition_penalty=rep_penalty,
            seed=seed
        )
        print(text)
        print("-" * 55)


def interactive_mode(model: TinyGPTLanguageModel, tokenizer):
    tok_type = "BPE Subwords" if isinstance(tokenizer, BPETokenizer) else "Characters"
    print("=" * 70)
    print(f"      STAGE 7: TINY GPT TRANSFORMER TEXT GENERATOR ({tok_type.upper()})")
    print("=" * 70)
    print(f"Model Configuration: {model.n_heads} Heads | d_model={model.d_model} | Vocab={model.vocab_size} {tok_type}")
    print("Educational Note: This model operates from scratch with causal self-attention.")
    print("Type 'q' or 'exit' anytime to quit.\n")

    while True:
        try:
            prompt = input("Enter prompt (e.g., 'The fox ', 'Once upon a time ', 'The little robot '): ").strip()
            if prompt.lower() in ("q", "quit", "exit"):
                print("Goodbye!")
                break
            if not prompt:
                prompt = "The "

            if isinstance(tokenizer, BPETokenizer):
                subwords = tokenizer.tokenize_to_strings(prompt)
                print(f"  [BPE Subword Decomposition: {subwords}]")

            unit_label = "tokens (~1.6 chars each)" if isinstance(tokenizer, BPETokenizer) else "characters"
            length_str = input(f"How many {unit_label} to generate? (default 50): ").strip()
            length = int(length_str) if length_str else 50

            temp_str = input("Temperature [0.3 = focused, 0.7 = balanced, 1.2 = wild] (default 0.7): ").strip()
            temp = float(temp_str) if temp_str else 0.7

            if length <= 0:
                print(">> Length must be greater than 0.\n")
                continue
            if temp <= 0:
                print(">> Temperature must be greater than 0.\n")
                continue

            print("\nGenerating with Causal Self-Attention...")
            print("-" * 60)
            generated = model.generate(
                tokenizer,
                prompt=prompt,
                max_new_tokens=length,
                temperature=temp,
                repetition_penalty=1.1
            )
            print(generated)
            print("-" * 60 + "\n")

        except ValueError as e:
            print(f">> Invalid input: {e}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(description="Generate text using the Tiny GPT Transformer.")
    parser.add_argument("--prompt", type=str, default=None, help="Initial text prompt (e.g. 'The fox ')")
    parser.add_argument("--length", type=int, default=50, help="Number of new tokens to generate (default: 50)")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--temp-sweep", action="store_true", help="Run a comparison across multiple temperatures")
    parser.add_argument("--rep-penalty", type=float, default=1.1, help="Repetition penalty factor (default: 1.1)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-K token filtering (optional)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-P nucleus sampling threshold (e.g. 0.9)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file path (.npz)")
    parser.add_argument("--vocab", type=str, default=None, help="Vocabulary JSON file path")
    args = parser.parse_args()

    model, tokenizer = load_transformer(checkpoint_path=args.checkpoint, vocab_path=args.vocab)

    if args.temp_sweep:
        prompt = args.prompt if args.prompt is not None else "The "
        run_temperature_sweep(
            model,
            tokenizer,
            prompt=prompt,
            length=args.length,
            rep_penalty=args.rep_penalty,
            seed=args.seed
        )
    elif args.prompt is not None:
        if isinstance(tokenizer, BPETokenizer):
            print(f"Prompt: {repr(args.prompt)}")
            print(f"BPE Subwords: {tokenizer.tokenize_to_strings(args.prompt)}")
            print("-" * 60)
        generated = model.generate(
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.length,
            temperature=args.temp,
            repetition_penalty=args.rep_penalty,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed
        )
        print(generated)
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
