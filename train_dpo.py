"""
train_dpo.py
------------
Stage 8: Direct Preference Optimization (DPO) from Scratch (Pure Python & NumPy).

What is Direct Preference Optimization (DPO)?
---------------------------------------------
In modern alignment (e.g. LLaMA-2/3, Claude, Zephyr), models are trained not just to imitate text
or follow instructions, but to prefer helpful, accurate completions (y_w) over unhelpful,
rambling, or incorrect completions (y_l).

Traditional RLHF requires training a complex separate reward model r(x, y) and using PPO.
DPO (Rafailov et al., NeurIPS 2023) mathematically proves that the optimal policy can be trained
DIRECTLY on preference pairs (x, y_w, y_l) using an exact closed-form loss:

    L_DPO = -log sigma( beta * [ (log pi_theta(y_w|x) - log pi_ref(y_w|x))
                               - (log pi_theta(y_l|x) - log pi_ref(y_l|x)) ] )

Where:
- pi_ref is a frozen copy of the reference model (the SFT model).
- pi_theta is the active trainable model.
- beta is the KL penalty / temperature hyperparameter (default: 0.1).
- Margin M = beta * ( (log pi_theta(y_w) - log pi_ref(y_w)) - (log pi_theta(y_l) - log pi_ref(y_l)) )
- Loss L_DPO = log(1 + exp(-M))

Analytical Gradients on Output Logits:
- Scale: w = beta * sigma(-M) = beta / (1 + exp(M))
- For chosen completion:   dZ_w = +w * (probs_w - 1_yw) / N_w
- For rejected completion: dZ_l = -w * (probs_l - 1_yl) / N_l

This script implements mathematically exact DPO without PyTorch or external RL libraries.
"""

import os
import json
import argparse
import numpy as np
from typing import List, Tuple, Dict
from src.tokenizer import BPETokenizer
from src.transformer_model import TinyGPTLanguageModel, AdamOptimizer, stable_softmax


def load_preference_dataset(filepath: str) -> List[Dict[str, str]]:
    """Loads JSONL preference dataset (prompt, chosen, rejected)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preference dataset '{filepath}' not found!")
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def tokenize_sequence(prompt_str: str, response_str: str, tokenizer: BPETokenizer, context_length: int = 16):
    """Encodes (prompt + response) and creates X and response-masked y."""
    prompt_tokens = tokenizer.encode(prompt_str, handle_unknown="replace")
    full_tokens = tokenizer.encode(prompt_str + response_str, handle_unknown="replace")
    prompt_len = len(prompt_tokens)

    # Use the last context_length tokens to align with model window
    if len(full_tokens) > context_length + 1:
        tokens_window = full_tokens[-(context_length + 1):]
        # Recalculate prompt boundary relative to window
        offset = len(full_tokens) - (context_length + 1)
        rel_prompt_len = max(prompt_len - offset, 0)
    else:
        pad_len = (context_length + 1) - len(full_tokens)
        tokens_window = [0] * pad_len + full_tokens
        rel_prompt_len = prompt_len + pad_len

    x_seq = np.array([tokens_window[:-1]], dtype=int)
    y_raw = tokens_window[1:]

    # Mask prompt tokens with -100
    y_masked = []
    for pos, tok in enumerate(y_raw):
        if pos < rel_prompt_len - 1 or tok == 0:
            y_masked.append(-100)
        else:
            y_masked.append(tok)

    y_seq = np.array([y_masked], dtype=int)
    return x_seq, y_seq


def main():
    parser = argparse.ArgumentParser(description="Direct Preference Optimization (DPO) for Tiny GPT Transformer.")
    parser.add_argument("--data", type=str, default="data/preference_data.jsonl", help="Path to preference dataset JSONL")
    parser.add_argument("--sft-checkpoint", type=str, default="transformer_sft_checkpoint.npz", help="Reference/initial checkpoint")
    parser.add_argument("--vocab", type=str, default="bpe_vocab.json", help="Vocabulary JSON file")
    parser.add_argument("--dpo-checkpoint", type=str, default="transformer_dpo_checkpoint.npz", help="Output DPO checkpoint")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature beta (default: 0.1)")
    parser.add_argument("--epochs", type=int, default=15, help="DPO training epochs (default: 15)")
    parser.add_argument("--lr", type=float, default=0.0005, help="DPO learning rate (default: 0.0005)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print("=" * 72)
    print("      STAGE 8: DIRECT PREFERENCE OPTIMIZATION (DPO) FROM SCRATCH")
    print("=" * 72)

    np.random.seed(args.seed)

    if not os.path.exists(args.vocab):
        raise FileNotFoundError(f"Vocabulary file '{args.vocab}' not found!")

    tokenizer = BPETokenizer()
    tokenizer.load(args.vocab)

    # Resolve initial checkpoint: use SFT checkpoint if it exists, else base checkpoint
    ckpt_path = args.sft_checkpoint
    if not os.path.exists(ckpt_path):
        ckpt_path = "transformer_bpe_checkpoint.npz"
        print(f"Notice: SFT checkpoint '{args.sft_checkpoint}' not found, falling back to base checkpoint '{ckpt_path}'.")

    # 1. Initialize Frozen Reference Model (pi_ref)
    ref_model = TinyGPTLanguageModel(vocab_size=tokenizer.vocab_size)
    ref_model.load_checkpoint(ckpt_path, expected_vocab_size=tokenizer.vocab_size)
    print(f"1. Loaded Frozen Reference Policy (pi_ref) from: {ckpt_path}")

    # 2. Initialize Active Trainable Model (pi_theta)
    model = TinyGPTLanguageModel(vocab_size=tokenizer.vocab_size)
    model.load_checkpoint(ckpt_path, expected_vocab_size=tokenizer.vocab_size)
    print(f"2. Initialized Active Policy (pi_theta) from:   {ckpt_path}")

    # 3. Load Preference Data
    items = load_preference_dataset(args.data)
    print(f"3. Loaded {len(items)} preference pairs from:   {args.data}")
    print(f"   DPO Beta: {args.beta} | Learning Rate: {args.lr} | Epochs: {args.epochs}\n")

    optimizer = AdamOptimizer(learning_rate=args.lr, weight_decay=0.01, grad_clip=1.0)

    print("-" * 72)
    print(f"{'Epoch':^7} | {'DPO Loss':^14} | {'Margin (M)':^14} | {'Accuracy':^12} | {'Status':^14}")
    print("-" * 72)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_margin = 0.0
        correct_count = 0

        # Shuffle pairs
        perm = np.random.permutation(len(items))

        for idx in perm:
            pair = items[idx]
            prompt = pair["prompt"]
            chosen = pair["chosen"]
            rejected = pair["rejected"]

            # Tokenize chosen and rejected sequences
            x_w, y_w = tokenize_sequence(prompt, chosen, tokenizer, context_length=model.context_length)
            x_l, y_l = tokenize_sequence(prompt, rejected, tokenizer, context_length=model.context_length)

            valid_w = (y_w != -100) & (y_w >= 0)
            valid_l = (y_l != -100) & (y_l >= 0)
            nw = max(int(np.sum(valid_w)), 1)
            nl = max(int(np.sum(valid_l)), 1)

            # 1. Log probabilities under frozen reference model
            log_ref_w = float(np.sum(ref_model.get_token_log_probs(x_w, y_w, ignore_index=-100))) / nw
            log_ref_l = float(np.sum(ref_model.get_token_log_probs(x_l, y_l, ignore_index=-100))) / nl

            # 2. Forward pass and log probabilities under active model
            _, probs_w, cache_w = model.forward(x_w)
            _, probs_l, cache_l = model.forward(x_l)

            # Calculate active log probs manually to use probabilities from cache
            batch_idx_w = np.arange(x_w.shape[0])[:, None]
            time_idx_w = np.arange(x_w.shape[1])[None, :]
            safe_yw = np.where(valid_w, y_w, 0)
            log_theta_w = float(np.sum(np.log(probs_w[batch_idx_w, time_idx_w, safe_yw] + 1e-15) * valid_w)) / nw

            batch_idx_l = np.arange(x_l.shape[0])[:, None]
            time_idx_l = np.arange(x_l.shape[1])[None, :]
            safe_yl = np.where(valid_l, y_l, 0)
            log_theta_l = float(np.sum(np.log(probs_l[batch_idx_l, time_idx_l, safe_yl] + 1e-15) * valid_l)) / nl

            # 3. Implicit Rewards and DPO Margin
            reward_w = args.beta * (log_theta_w - log_ref_w)
            reward_l = args.beta * (log_theta_l - log_ref_l)
            margin = reward_w - reward_l

            # 4. DPO Loss: log(1 + exp(-margin))
            loss = np.log1p(np.exp(-np.clip(margin, -20.0, 20.0)))
            epoch_loss += loss
            epoch_margin += margin
            if reward_w > reward_l:
                correct_count += 1

            # 5. Analytical Logit Gradients
            # Scale w = beta * sigma(-margin)
            scale = args.beta / (1.0 + np.exp(np.clip(margin, -20.0, 20.0)))

            # Chosen gradient: +scale * (p_w - 1_yw) / nw
            dZw = scale * probs_w.copy()
            dZw[batch_idx_w, time_idx_w, safe_yw] -= scale
            dZw *= valid_w[:, :, None]
            dZw /= nw

            # Rejected gradient: -scale * (p_l - 1_yl) / nl
            dZl = -scale * probs_l.copy()
            dZl[batch_idx_l, time_idx_l, safe_yl] += scale
            dZl *= valid_l[:, :, None]
            dZl /= nl

            # 6. Backpropagation
            grads_w = model.backward_from_dlogits(dZw, cache_w)
            grads_l = model.backward_from_dlogits(dZl, cache_l)

            # Accumulate gradients from both completions
            total_grads = {k: grads_w[k] + grads_l[k] for k in grads_w}
            optimizer.step(model, total_grads)

        avg_loss = epoch_loss / len(items)
        avg_margin = epoch_margin / len(items)
        acc_pct = (correct_count / len(items)) * 100.0

        status = "Aligning" if acc_pct < 100.0 else "Aligned"
        print(f"{epoch:^7} | {avg_loss:^14.4f} | {avg_margin:^14.4f} | {acc_pct:^11.1f}% | {status:<14}")

    print("-" * 72)

    # Save DPO model checkpoint
    model.save_checkpoint(args.dpo_checkpoint)
    print(f"\nDPO Aligned Checkpoint saved to: {args.dpo_checkpoint}")

    # Comparative evaluation
    print("\n" + "=" * 72)
    print("      DPO PREFERENCE COMPARISON")
    print("=" * 72)
    test_sample = items[0]
    prompt = test_sample["prompt"]
    print(f"Prompt:\n{prompt}")

    gen_ref = ref_model.generate(tokenizer, prompt=prompt, max_new_tokens=25, temperature=0.6, top_p=0.9, seed=42)
    gen_dpo = model.generate(tokenizer, prompt=prompt, max_new_tokens=25, temperature=0.6, top_p=0.9, seed=42)

    print(f"Reference Policy Output:\n  {repr(gen_ref[len(prompt):].strip())}\n")
    print(f"DPO Aligned Policy Output:\n  {repr(gen_dpo[len(prompt):].strip())}\n")
    print("=" * 72)


if __name__ == "__main__":
    main()

