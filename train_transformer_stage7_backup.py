"""
train_transformer.py
--------------------
Stage 7: Trains the Tiny GPT Transformer on clean English text with:
1. Byte-Pair Encoding (BPE) subword tokenization (or optional character tokenization).
2. Sequential Train (85%) / Validation (15%) split (strictly causal, zero data leakage).
3. Decoupled L2 Weight Decay (AdamW) regularizing 2D weight matrices.
4. Validation Loss monitoring every epoch.
5. Early Stopping with configurable patience.
6. Best-Checkpoint Saving: Saves the model with the LOWEST validation loss.
7. Fixed random seed (42) for reproducible training.
"""

import os
import argparse
from typing import Optional
import numpy as np
from src.tokenizer import CharTokenizer, BPETokenizer
from src.transformer_model import TinyGPTLanguageModel, AdamOptimizer


def prepare_sequences(encoded_tokens: np.ndarray, context_length: int, step: int = 3):
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
        raise ValueError(f"Token stream length ({len(encoded_tokens)}) is too short for context {context_length}!")

    return np.array(X_list, dtype=int), np.array(y_list, dtype=int)


def evaluate_loss(model: TinyGPTLanguageModel, X_val: np.ndarray, y_val: np.ndarray, batch_size: int = 32) -> float:
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


def save_best_weights(model: TinyGPTLanguageModel) -> dict:
    """Creates a deep in-memory copy of all model parameters."""
    return {
        "W_te": model.W_te.copy(),
        "W_pe": model.W_pe.copy(),
        "ln1_gamma": model.block.ln1.gamma.copy(),
        "ln1_beta": model.block.ln1.beta.copy(),
        "Wq": model.block.attn.Wq.copy(),
        "bq": model.block.attn.bq.copy(),
        "Wk": model.block.attn.Wk.copy(),
        "bk": model.block.attn.bk.copy(),
        "Wv": model.block.attn.Wv.copy(),
        "bv": model.block.attn.bv.copy(),
        "Wo": model.block.attn.Wo.copy(),
        "bo": model.block.attn.bo.copy(),
        "ln2_gamma": model.block.ln2.gamma.copy(),
        "ln2_beta": model.block.ln2.beta.copy(),
        "W1": model.block.ffn.W1.copy(),
        "b1": model.block.ffn.b1.copy(),
        "W2": model.block.ffn.W2.copy(),
        "b2": model.block.ffn.b2.copy(),
        "ln_f_gamma": model.ln_f.gamma.copy(),
        "ln_f_beta": model.ln_f.beta.copy(),
        "W_lm": model.W_lm.copy(),
        "b_lm": model.b_lm.copy(),
    }


def restore_best_weights(model: TinyGPTLanguageModel, state_dict: dict):
    """Restores parameters from in-memory copy."""
    model.W_te = state_dict["W_te"].copy()
    model.W_pe = state_dict["W_pe"].copy()
    model.block.ln1.gamma = state_dict["ln1_gamma"].copy()
    model.block.ln1.beta = state_dict["ln1_beta"].copy()
    model.block.attn.Wq = state_dict["Wq"].copy()
    model.block.attn.bq = state_dict["bq"].copy()
    model.block.attn.Wk = state_dict["Wk"].copy()
    model.block.attn.bk = state_dict["bk"].copy()
    model.block.attn.Wv = state_dict["Wv"].copy()
    model.block.attn.bv = state_dict["bv"].copy()
    model.block.attn.Wo = state_dict["Wo"].copy()
    model.block.attn.bo = state_dict["bo"].copy()
    model.block.ln2.gamma = state_dict["ln2_gamma"].copy()
    model.block.ln2.beta = state_dict["ln2_beta"].copy()
    model.block.ffn.W1 = state_dict["W1"].copy()
    model.block.ffn.b1 = state_dict["b1"].copy()
    model.block.ffn.W2 = state_dict["W2"].copy()
    model.block.ffn.b2 = state_dict["b2"].copy()
    model.ln_f.gamma = state_dict["ln_f_gamma"].copy()
    model.ln_f.beta = state_dict["ln_f_beta"].copy()
    model.W_lm = state_dict["W_lm"].copy()
    model.b_lm = state_dict["b_lm"].copy()


def train(
    dataset_path: str = "data/clean_corpus.txt",
    vocab_save_path: Optional[str] = None,
    checkpoint_save_path: Optional[str] = None,
    tokenizer_type: str = "bpe",
    num_merges: int = 60,
    max_epochs: int = 50,
    patience: int = 6,
    learning_rate: float = 0.003,
    weight_decay: float = 0.01,
    context_length: int = 16,
    d_model: int = 32,
    n_heads: int = 2,
    d_ff: int = 64,
    batch_size: int = 32,
    seed: int = 42
):
    print("=" * 72)
    print("      STAGE 7: TRAINING TRANSFORMER WITH BPE SUBWORD TOKENIZATION")
    print("=" * 72)

    # Resolve default paths
    if vocab_save_path is None:
        vocab_save_path = "bpe_vocab.json" if tokenizer_type == "bpe" else "transformer_vocab.json"
    if checkpoint_save_path is None:
        checkpoint_save_path = "transformer_bpe_checkpoint.npz" if tokenizer_type == "bpe" else "transformer_checkpoint.npz"

    # Fixed seed for reproducibility
    np.random.seed(seed)

    if not os.path.exists(dataset_path):
        fallback = os.path.join("data", "tiny_story.txt")
        print(f"Notice: '{dataset_path}' not found, falling back to '{fallback}'.")
        dataset_path = fallback

    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Fit Tokenizer
    if tokenizer_type == "bpe":
        tokenizer = BPETokenizer(unk_token="<unk>")
        tokenizer.fit(text, num_merges=num_merges, min_freq=2)
        tokenizer.save(vocab_save_path)
        encoded = np.array(tokenizer.encode(text), dtype=int)
        vocab_size = tokenizer.vocab_size
        compression = len(text) / max(len(encoded), 1)
        savings = (1.0 - len(encoded) / len(text)) * 100

        print(f"1. Dataset & BPE Subword Tokenizer Details:")
        print(f"   - Source file:          {dataset_path}")
        print(f"   - Raw characters:       {len(text):,}")
        print(f"   - Base characters:      {len(tokenizer.base_vocab)} unique")
        print(f"   - Learned BPE merges:   {len(tokenizer.merges_list)}")
        print(f"   - Final vocabulary:     {vocab_size} subwords")
        print(f"   - Compressed tokens:    {len(encoded):,} tokens")
        print(f"   - Compression ratio:    {compression:.2f}x ({savings:.1f}% sequence length reduction)")
        print(f"   - Top 5 learned merges: {[tokenizer.id_to_token[nid] for _, nid in tokenizer.merges_list[:5]]}")
    else:
        tokenizer = CharTokenizer(unk_token="<unk>")
        tokenizer.fit(text)
        tokenizer.save(vocab_save_path)
        encoded = np.array(tokenizer.encode(text), dtype=int)
        vocab_size = tokenizer.vocab_size

        print(f"1. Dataset & Character Tokenizer Details:")
        print(f"   - Source file:          {dataset_path}")
        print(f"   - Total characters:     {len(text):,}")
        print(f"   - Vocabulary size:      {vocab_size} unique characters")

    # 2. Sequential Train (85%) / Validation (15%) Split
    split_idx = int(len(encoded) * 0.85)
    train_tokens = encoded[:split_idx]
    val_tokens = encoded[split_idx:]

    print(f"   - Training split:       {len(train_tokens):,} tokens (first 85%)")
    print(f"   - Validation split:     {len(val_tokens):,} tokens (last 15% - unseen)")

    # 3. Prepare sequences
    X_train, y_train = prepare_sequences(train_tokens, context_length=context_length, step=3)
    X_val, y_val = prepare_sequences(val_tokens, context_length=context_length, step=3)

    print(f"   - Training batches:     {len(X_train):,} sequences of length {context_length}")
    print(f"   - Validation batches:   {len(X_val):,} sequences of length {context_length}")

    # 4. Model & Optimizer Configuration
    print(f"\n2. Training & Regularization Settings:")
    print(f"   - Context Window (T):       {context_length} tokens (~{context_length * (len(text)/len(encoded)):.0f} characters)")
    print(f"   - Embedding Dim (d_model):   {d_model}")
    print(f"   - Attention Heads:          {n_heads} (head dimension = {d_model // n_heads})")
    print(f"   - Feed-Forward Dim (d_ff):   {d_ff}")
    print(f"   - Max Epochs:               {max_epochs}")
    print(f"   - Early Stopping Patience:  {patience} epochs")
    print(f"   - L2 Weight Decay:          {weight_decay} (AdamW decoupled decay on 2D weights)")
    print(f"   - Learning Rate:            {learning_rate}")
    print(f"   - Batch Size:               {batch_size}")
    print(f"   - Random Seed:              {seed}")

    baseline_loss = -np.log(1.0 / vocab_size)
    print(f"   - Theoretical Baseline Loss (-log(1/{vocab_size})): {baseline_loss:.4f}\n")

    model = TinyGPTLanguageModel(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff
    )

    optimizer = AdamOptimizer(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=1.0
    )

    # Initial losses
    initial_train_loss = evaluate_loss(model, X_train, y_train, batch_size=batch_size)
    initial_val_loss = evaluate_loss(model, X_val, y_val, batch_size=batch_size)
    print(f"Initial State:")
    print(f"  Train Loss: {initial_train_loss:.4f} | Val Loss: {initial_val_loss:.4f}\n")

    best_val_loss = float("inf")
    best_epoch = 0
    best_weights_snapshot = None
    best_train_loss_at_best_epoch = initial_train_loss
    patience_counter = 0

    num_train_samples = len(X_train)

    print("-" * 72)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^12} | {'Best Val':^12} | {'Status':^18}")
    print("-" * 72)

    for epoch in range(1, max_epochs + 1):
        indices = np.random.permutation(num_train_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, num_train_samples, batch_size):
            x_b = X_shuffled[i : i + batch_size]
            y_b = y_shuffled[i : i + batch_size]

            _, probs, cache = model.forward(x_b)
            loss = model.compute_loss(probs, y_b)
            if not np.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered at epoch {epoch}: {loss}")

            epoch_loss += loss * len(x_b)
            num_batches += len(x_b)

            grads = model.backward(y_b, cache)
            optimizer.step(model, grads)

        avg_train_loss = epoch_loss / max(num_batches, 1)
        val_loss = evaluate_loss(model, X_val, y_val, batch_size=batch_size)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_train_loss_at_best_epoch = avg_train_loss
            patience_counter = 0
            best_weights_snapshot = save_best_weights(model)
            status = f"* NEW BEST ({val_loss:.4f})"
        else:
            patience_counter += 1
            status = f"Patience {patience_counter}/{patience}"

        print(f"{epoch:^7} | {avg_train_loss:^12.4f} | {val_loss:^12.4f} | {best_val_loss:^12.4f} | {status:<18}")

        if patience_counter >= patience:
            print("-" * 72)
            print(f"Early stopping triggered at Epoch {epoch}! No validation improvement for {patience} epochs.")
            break

    print("-" * 72)

    # Restore best checkpoint
    print(f"\nRestoring model weights from Best Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})...")
    if best_weights_snapshot is not None:
        restore_best_weights(model, best_weights_snapshot)

    # Save checkpoint
    model.save_checkpoint(checkpoint_save_path)
    print(f"Saved best model checkpoint to: {checkpoint_save_path}")

    # Final verification of restored weights
    restored_val_loss = evaluate_loss(model, X_val, y_val, batch_size=batch_size)
    assert abs(restored_val_loss - best_val_loss) < 1e-4, "Restored weights loss mismatch!"

    val_improved = best_val_loss < initial_val_loss
    improvement_pct = ((initial_val_loss - best_val_loss) / initial_val_loss) * 100.0

    print("\n" + "=" * 72)
    print("      STAGE 7 TRAINING & GENERALIZATION REPORT")
    print("=" * 72)
    print(f"1. Tokenizer Type:          {tokenizer_type.upper()} ({vocab_size} tokens)")
    print(f"2. Dataset Size:            {len(text):,} characters -> {len(encoded):,} tokens")
    print(f"3. Best Epoch:              Epoch {best_epoch} of {max_epochs}")
    print(f"4. Initial Validation Loss: {initial_val_loss:.4f}")
    print(f"5. Best Validation Loss:    {best_val_loss:.4f} ({'IMPROVED' if val_improved else 'DID NOT IMPROVE'})")
    print(f"6. Val Loss Improvement:    {improvement_pct:+.2f}%")
    print(f"7. Train Loss at Best Epoch:{best_train_loss_at_best_epoch:.4f}")
    print(f"8. Checkpoint Saved:        '{checkpoint_save_path}' (Contains Epoch {best_epoch} weights)")
    print(f"9. Vocabulary Saved:        '{vocab_save_path}'")
    print("=" * 72 + "\n")

    print("Sample Generations from Best Checkpoint:")
    print("-" * 65)
    for temp in [0.4, 0.7, 1.0]:
        gen = model.generate(tokenizer, prompt="The ", max_new_tokens=40, temperature=temp, repetition_penalty=1.1, seed=seed)
        print(f"[Temperature = {temp}]")
        print(repr(gen))
        print()
    print("-" * 65)


def main():
    parser = argparse.ArgumentParser(description="Train Tiny GPT with BPE Subword Tokenization.")
    parser.add_argument("--tokenizer", type=str, default="bpe", choices=["bpe", "char"], help="Tokenizer type: 'bpe' or 'char'")
    parser.add_argument("--num-merges", type=int, default=60, help="Number of BPE merges (default: 60)")
    parser.add_argument("--dataset", type=str, default="data/clean_corpus.txt", help="Path to cleaned dataset text")
    parser.add_argument("--vocab", type=str, default=None, help="Path to save tokenizer vocab JSON")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to save model checkpoint .npz")
    parser.add_argument("--epochs", type=int, default=35, help="Maximum epochs to train (default: 35)")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (default: 6)")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate (default: 0.003)")
    parser.add_argument("--decay", type=float, default=0.01, help="L2 weight decay factor (default: 0.01)")
    parser.add_argument("--context-length", type=int, default=16, help="Context length in tokens (default: 16)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        vocab_save_path=args.vocab,
        checkpoint_save_path=args.checkpoint,
        tokenizer_type=args.tokenizer,
        num_merges=args.num_merges,
        max_epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr,
        weight_decay=args.decay,
        context_length=args.context_length,
        batch_size=args.batch_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
