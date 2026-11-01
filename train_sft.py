"""
train_sft.py
------------
Stage 8: Supervised Fine-Tuning (SFT / Instruction Tuning) from Scratch.

What is Supervised Fine-Tuning (SFT)?
-------------------------------------
Pre-training (Stage 7) teaches the model language structure, subwords, and grammar by predicting
the next token in raw prose. However, pre-trained base models do not know how to follow instructions
or act as an assistant—they merely continue whatever text was provided.

Instruction Tuning (SFT) fine-tunes the pre-trained base model on structured pairs of:
    ### Instruction:\n{instruction}\n\n### Response:\n{response}

Key Reliability Technique: Response-Only Loss Masking:
During SFT, the model's loss and gradients are computed ONLY on the {response} tokens.
Prompt positions are masked with target = -100 so the model is not penalized for the words
in the user's question, focusing 100% of its learning capacity on answering!

This script:
1. Loads the pre-trained base checkpoint (transformer_bpe_checkpoint.npz) and vocabulary (bpe_vocab.json).
2. Loads instruction-response examples from 'data/instruction_data.jsonl'.
3. Formats instructions and applies response-only loss masking (target = -100 on prompt tokens).
4. Fine-tunes the model using decoupled AdamW weight decay.
5. Preserves the base model checkpoint and saves SFT weights to 'transformer_sft_checkpoint.npz'.
6. Evaluates on held-out test instructions and compares base-model vs SFT-model outputs.
"""

import os
import json
import argparse
import numpy as np
from typing import List, Tuple, Dict
from src.tokenizer import BPETokenizer
from src.transformer_model import TinyGPTLanguageModel, AdamOptimizer


def load_instruction_dataset(filepath: str) -> List[Dict[str, str]]:
    """Loads JSONL instruction-response dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Instruction dataset '{filepath}' not found!")
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def format_and_tokenize_sft(
    items: List[Dict[str, str]],
    tokenizer: BPETokenizer,
    context_length: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Formats instruction examples and prepares X, y pairs.
    Applies response-only loss masking: prompt targets are set to -100.
    """
    X_list = []
    y_list = []

    for item in items:
        prompt_str = f"Instruction:\n{item['instruction']}\n\nResponse:\n"
        resp_str = f"{item['response']}\n"

        prompt_tokens = tokenizer.encode(prompt_str, handle_unknown="replace")
        full_tokens = tokenizer.encode(prompt_str + resp_str, handle_unknown="replace")

        prompt_len = len(prompt_tokens)
        total_len = len(full_tokens)

        # Slide a context window across the sequence
        for i in range(0, total_len - 1, 2):
            end_idx = min(i + context_length, total_len - 1)
            x_seq = full_tokens[i : end_idx]
            y_seq = full_tokens[i + 1 : end_idx + 1]

            # Pad if shorter than context_length
            if len(x_seq) < context_length:
                pad_len = context_length - len(x_seq)
                x_seq = x_seq + [0] * pad_len
                y_seq = y_seq + [-100] * pad_len

            # Mask out prompt target tokens
            y_masked = []
            for pos_in_window, tok in enumerate(y_seq):
                abs_pos = i + 1 + pos_in_window
                if abs_pos < prompt_len or tok == -100:
                    y_masked.append(-100)
                else:
                    y_masked.append(tok)

            # Only include windows that have at least one valid response token to predict
            if any(t != -100 for t in y_masked):
                X_list.append(x_seq)
                y_list.append(y_masked)

    if not X_list:
        raise ValueError("No valid SFT training windows could be constructed!")

    return np.array(X_list, dtype=int), np.array(y_list, dtype=int)


def evaluate_sft_loss(model: TinyGPTLanguageModel, X: np.ndarray, y: np.ndarray, batch_size: int = 16) -> float:
    """Evaluates cross-entropy loss strictly on response tokens."""
    total_loss = 0.0
    num_samples = len(X)
    valid_batches = 0

    for i in range(0, num_samples, batch_size):
        xb = X[i : i + batch_size]
        yb = y[i : i + batch_size]
        _, probs, _ = model.forward(xb)
        loss = model.compute_loss(probs, yb, ignore_index=-100)
        if np.isfinite(loss):
            total_loss += loss
            valid_batches += 1

    return total_loss / max(valid_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) for Tiny GPT Transformer.")
    parser.add_argument("--data", type=str, default="data/instruction_data.jsonl", help="Path to instruction JSONL")
    parser.add_argument("--base-checkpoint", type=str, default="transformer_bpe_checkpoint.npz", help="Base checkpoint")
    parser.add_argument("--vocab", type=str, default="bpe_vocab.json", help="Vocabulary JSON file")
    parser.add_argument("--sft-checkpoint", type=str, default="transformer_sft_checkpoint.npz", help="Output SFT checkpoint")
    parser.add_argument("--epochs", type=int, default=20, help="SFT epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=0.001, help="SFT learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print("=" * 72)
    print("      STAGE 8: SUPERVISED FINE-TUNING (SFT / INSTRUCTION TUNING)")
    print("=" * 72)

    np.random.seed(args.seed)

    # 1. Load Tokenizer & Base Checkpoint
    if not os.path.exists(args.vocab):
        raise FileNotFoundError(f"Vocabulary '{args.vocab}' not found! Run train_transformer.py first.")
    if not os.path.exists(args.base_checkpoint):
        raise FileNotFoundError(f"Base checkpoint '{args.base_checkpoint}' not found!")

    tokenizer = BPETokenizer()
    tokenizer.load(args.vocab)

    model = TinyGPTLanguageModel(vocab_size=tokenizer.vocab_size)
    model.load_checkpoint(args.base_checkpoint, expected_vocab_size=tokenizer.vocab_size)

    # 2. Load Instruction Dataset
    items = load_instruction_dataset(args.data)
    print(f"1. Dataset Details:")
    print(f"   - File:                {args.data}")
    print(f"   - Total Instructions:  {len(items)} examples")

    # Split: First 12 for training, last 3 held-out for validation
    split_idx = max(len(items) - 3, 1)
    train_items = items[:split_idx]
    val_items = items[split_idx:]

    print(f"   - Training examples:   {len(train_items)}")
    print(f"   - Validation examples: {len(val_items)} (held-out)")

    # 3. Format and Tokenize with Response-Only Masking
    X_train, y_train = format_and_tokenize_sft(train_items, tokenizer, context_length=model.context_length)
    X_val, y_val = format_and_tokenize_sft(val_items, tokenizer, context_length=model.context_length)

    print(f"   - Training windows:    {len(X_train)} (Response tokens only, prompt masked to -100)")
    print(f"   - Validation windows:  {len(X_val)}")

    # Baseline Losses before SFT
    init_train_loss = evaluate_sft_loss(model, X_train, y_train, batch_size=args.batch_size)
    init_val_loss = evaluate_sft_loss(model, X_val, y_val, batch_size=args.batch_size)
    print(f"\n2. Pre-Fine-Tuning State (Base Model):")
    print(f"   - Base Model Train SFT Loss: {init_train_loss:.4f}")
    print(f"   - Base Model Val SFT Loss:   {init_val_loss:.4f}\n")

    # 4. Fine-Tune with AdamW
    optimizer = AdamOptimizer(learning_rate=args.lr, weight_decay=0.01, grad_clip=1.0)
    best_val_loss = float("inf")
    best_weights = None

    print("-" * 65)
    print(f"{'Epoch':^7} | {'Train Loss':^14} | {'Val Loss':^14} | {'Status':^20}")
    print("-" * 65)

    n_samples = len(X_train)
    for epoch in range(1, args.epochs + 1):
        perm = np.random.permutation(n_samples)
        X_shuff = X_train[perm]
        y_shuff = y_train[perm]

        epoch_loss = 0.0
        n_batches = 0

        for b in range(0, n_samples, args.batch_size):
            xb = X_shuff[b : b + args.batch_size]
            yb = y_shuff[b : b + args.batch_size]

            _, probs, cache = model.forward(xb)
            loss = model.compute_loss(probs, yb, ignore_index=-100)
            if np.isfinite(loss):
                epoch_loss += loss * len(xb)
                n_batches += len(xb)

            grads = model.backward(yb, cache, ignore_index=-100)
            optimizer.step(model, grads)

        avg_train = epoch_loss / max(n_batches, 1)
        cur_val = evaluate_sft_loss(model, X_val, y_val, batch_size=args.batch_size)

        if cur_val < best_val_loss:
            best_val_loss = cur_val
            from train_transformer import save_best_weights, restore_best_weights
            best_weights = save_best_weights(model)
            status = f"* NEW BEST ({cur_val:.4f})"
        else:
            status = "-"

        print(f"{epoch:^7} | {avg_train:^14.4f} | {cur_val:^14.4f} | {status:<20}")

    print("-" * 65)

    # 5. Restore Best SFT Checkpoint
    if best_weights is not None:
        restore_best_weights(model, best_weights)

    model.save_checkpoint(args.sft_checkpoint)
    print(f"\nSFT Checkpoint saved to: {args.sft_checkpoint}")
    print(f"Base Checkpoint preserved: {args.base_checkpoint}")

    # 6. Comparative Evaluation on Held-Out Prompts
    print("\n" + "=" * 72)
    print("      HELD-OUT INSTRUCTION EVALUATION (BASE VS SFT)")
    print("=" * 72)

    # Load base model for comparison
    base_model = TinyGPTLanguageModel(vocab_size=tokenizer.vocab_size)
    base_model.load_checkpoint(args.base_checkpoint, expected_vocab_size=tokenizer.vocab_size)

    for idx, test_item in enumerate(val_items, 1):
        prompt = f"Instruction:\n{test_item['instruction']}\n\nResponse:\n"
        print(f"\n[Test Example {idx}]")
        print(f"Instruction:      {repr(test_item['instruction'])}")
        print(f"Target Response:  {repr(test_item['response'])}")

        base_out = base_model.generate(
            tokenizer, prompt=prompt, max_new_tokens=25, temperature=0.6, top_p=0.9, seed=42
        )
        sft_out = model.generate(
            tokenizer, prompt=prompt, max_new_tokens=25, temperature=0.6, top_p=0.9, seed=42
        )

        base_resp = base_out.split("Response:\n")[-1].strip()
        sft_resp = sft_out.split("Response:\n")[-1].strip()

        print(f"Base Model Output: {repr(base_resp)}")
        print(f"SFT Model Output:  {repr(sft_resp)}")

    print("\n" + "=" * 72)
    print("      SFT TRAINING COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()

