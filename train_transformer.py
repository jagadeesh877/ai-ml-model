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


def get_scheduled_lr(epoch: int, total_epochs: int, base_lr: float, warmup_epochs: int = 3, min_lr_ratio: float = 0.1) -> float:
    """
    Computes learning rate with linear warmup and cosine annealing.
    Warmup stabilizes initial random weights; cosine decay enables fine convergence.
    """
    if epoch <= warmup_epochs:
        return base_lr * (epoch / max(warmup_epochs, 1))
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    decay = 0.5 * (1.0 + np.cos(np.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * decay)


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
    checkpoint_dir: str = "checkpoints",
    tokenizer_type: str = "bpe",
    num_merges: int = 180,
    max_epochs: int = 5,
    patience: int = 0,
    learning_rate: float = 0.003,
    weight_decay: float = 0.01,
    context_length: int = 32,
    d_model: int = 64,
    n_heads: int = 4,
    d_ff: int = 128,
    batch_size: int = 32,
    step: int = 2,
    seed: int = 42
):
    print("=" * 72)
    print("      STAGE 8+: TRAINING TRANSFORMER WITH PER-EPOCH CHECKPOINTS")
    print("=" * 72)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resolve default paths
    if vocab_save_path is None:
        vocab_save_path = "bpe_vocab.json" if tokenizer_type == "bpe" else "transformer_vocab.json"

    # Fixed seed for reproducibility
    np.random.seed(seed)

    if not os.path.exists(dataset_path):
        fallback = os.path.join("data", "tiny_story.txt")
        print(f"Notice: '{dataset_path}' not found, falling back to '{fallback}'.")
        dataset_path = fallback

    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Strictly Sequential Train (85%) / Validation (15%) Split (Zero Data Leakage)
    split_idx = int(len(text) * 0.85)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # 2. Fit Tokenizer strictly on TRAINING split only (No validation leakage!)
    if tokenizer_type == "bpe":
        tokenizer = BPETokenizer(unk_token="<unk>")
        tokenizer.fit(train_text, num_merges=num_merges, min_freq=2)
        tokenizer.save(vocab_save_path)
        train_tokens = np.array(tokenizer.encode(train_text), dtype=int)
        val_tokens = np.array(tokenizer.encode(val_text, handle_unknown="replace"), dtype=int)
        vocab_size = tokenizer.vocab_size
        compression = len(train_text) / max(len(train_tokens), 1)
        savings = (1.0 - len(train_tokens) / len(train_text)) * 100

        print(f"1. Dataset & Leading-Space BPE Subword Tokenizer (Strict Train-Only Fitting):")
        print(f"   - Source file:          {dataset_path}")
        print(f"   - Total characters:     {len(text):,} (Train: {len(train_text):,}, Val: {len(val_text):,})")
        print(f"   - Base characters:      {len(tokenizer.base_vocab)} unique in training set")
        print(f"   - Learned BPE merges:   {len(tokenizer.merges_list)} (fitted strictly on training set)")
        print(f"   - Final vocabulary:     {vocab_size} subwords")
        print(f"   - Training tokens:      {len(train_tokens):,} tokens")
        print(f"   - Validation tokens:    {len(val_tokens):,} tokens (unseen)")
        print(f"   - Compression ratio:    {compression:.2f}x ({savings:.1f}% sequence reduction)")
        words_with_space = [w for w in tokenizer.vocab if len(w) > 2 and w.startswith(" ")]
        print(f"   - Whole words captured: {len(words_with_space)} (e.g. {words_with_space[:10]})")
    else:
        tokenizer = CharTokenizer(unk_token="<unk>")
        tokenizer.fit(train_text)
        tokenizer.save(vocab_save_path)
        train_tokens = np.array(tokenizer.encode(train_text), dtype=int)
        val_tokens = np.array(tokenizer.encode(val_text, handle_unknown="replace"), dtype=int)
        vocab_size = tokenizer.vocab_size

        print(f"1. Dataset & Character Tokenizer Details:")
        print(f"   - Source file:          {dataset_path}")
        print(f"   - Training split:       {len(train_tokens):,} characters (first 85%)")
        print(f"   - Validation split:     {len(val_tokens):,} characters (last 15% - unseen)")
        print(f"   - Vocabulary size:      {vocab_size} unique characters")

    # 3. Prepare sequences with configurable stride step
    X_train, y_train = prepare_sequences(train_tokens, context_length=context_length, step=step)
    X_val, y_val = prepare_sequences(val_tokens, context_length=context_length, step=step)

    print(f"   - Sequence stride (step): {step} tokens")
    print(f"   - Training batches:     {len(X_train):,} sequences of length {context_length}")
    print(f"   - Validation batches:   {len(X_val):,} sequences of length {context_length}")

    model = TinyGPTLanguageModel(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff
    )

    total_params = (
        model.W_te.size + model.W_pe.size +
        model.block.ln1.gamma.size + model.block.ln1.beta.size +
        model.block.attn.Wq.size + model.block.attn.bq.size +
        model.block.attn.Wk.size + model.block.attn.bk.size +
        model.block.attn.Wv.size + model.block.attn.bv.size +
        model.block.attn.Wo.size + model.block.attn.bo.size +
        model.block.ln2.gamma.size + model.block.ln2.beta.size +
        model.block.ffn.W1.size + model.block.ffn.b1.size +
        model.block.ffn.W2.size + model.block.ffn.b2.size +
        model.ln_f.gamma.size + model.ln_f.beta.size +
        model.W_lm.size + model.b_lm.size
    )

    total_tokens = len(train_tokens) + len(val_tokens)
    print(f"\n2. Scaled Architecture & Regularization Settings:")
    print(f"   - Total Parameters:         {total_params:,} weights")
    print(f"   - Context Window (T):       {context_length} tokens (~{context_length * (len(text)/total_tokens):.0f} characters)")
    print(f"   - Embedding Dim (d_model):   {d_model}")
    print(f"   - Attention Heads:          {n_heads} (head dimension = {d_model // n_heads})")
    print(f"   - Feed-Forward Dim (d_ff):   {d_ff}")
    print(f"   - Max Epochs:               {max_epochs}")
    print(f"   - Early Stopping Patience:  {patience if patience > 0 else 'Disabled (full run)'}")
    print(f"   - L2 Weight Decay:          {weight_decay} (AdamW decoupled decay on 2D weights)")
    print(f"   - Base Learning Rate:       {learning_rate} (Warmup + Cosine Annealing)")
    print(f"   - Batch Size:               {batch_size}")
    print(f"   - Random Seed:              {seed}")
    print(f"   - Checkpoint Directory:     '{checkpoint_dir}'")

    baseline_loss = -np.log(1.0 / vocab_size)
    print(f"   - Theoretical Baseline Loss (-log(1/{vocab_size})): {baseline_loss:.4f}\n")

    optimizer = AdamOptimizer(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=1.0
    )

    # Initial losses
    initial_train_loss = evaluate_loss(model, X_train, y_train, batch_size=batch_size)
    initial_val_loss = evaluate_loss(model, X_val, y_val, batch_size=batch_size)
    print(f"Initial State:")
    print(f"  Train Loss: {initial_train_loss:.4f} (Perplexity: {np.exp(initial_train_loss):.2f}) | "
          f"Val Loss: {initial_val_loss:.4f} (Perplexity: {np.exp(initial_val_loss):.2f})\n")

    best_val_loss = float("inf")
    best_epoch = 0
    best_weights_snapshot = None
    best_train_loss_at_best_epoch = initial_train_loss
    patience_counter = 0
    history = []

    num_train_samples = len(X_train)

    print("-" * 78)
    print(f"{'Epoch':^7} | {'LR':^8} | {'Train Loss':^12} | {'Val Loss':^12} | {'Best Val':^12} | {'Status':^18}")
    print("-" * 78)

    warmup_ep = min(2, max_epochs)
    for epoch in range(1, max_epochs + 1):
        cur_lr = get_scheduled_lr(epoch, max_epochs, learning_rate, warmup_epochs=warmup_ep)
        optimizer.lr = cur_lr

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

        # Save per-epoch checkpoint
        epoch_ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.npz")
        model.save_checkpoint(epoch_ckpt_path)

        epoch_record = {
            "epoch": epoch,
            "lr": cur_lr,
            "train_loss": round(float(avg_train_loss), 4),
            "val_loss": round(float(val_loss), 4),
            "train_ppl": round(float(np.exp(avg_train_loss)), 2),
            "val_ppl": round(float(np.exp(val_loss)), 2),
            "checkpoint": epoch_ckpt_path
        }
        history.append(epoch_record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_train_loss_at_best_epoch = avg_train_loss
            patience_counter = 0
            best_weights_snapshot = save_best_weights(model)
            best_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_best_val.npz")
            model.save_checkpoint(best_ckpt_path)
            status = f"* NEW BEST ({val_loss:.4f})"
        else:
            patience_counter += 1
            if patience > 0:
                status = f"Patience {patience_counter}/{patience}"
            else:
                status = f"Val +{val_loss - best_val_loss:.4f}"

        print(f"{epoch:^7} | {cur_lr:^8.5f} | {avg_train_loss:^12.4f} | {val_loss:^12.4f} | {best_val_loss:^12.4f} | {status:<18}")

        if patience > 0 and patience_counter >= patience:
            print("-" * 78)
            print(f"Early stopping triggered at Epoch {epoch}! No validation improvement for {patience} epochs.")
            break

    print("-" * 78)

    # Save training history JSON
    import json
    history_json_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved complete training history to: '{history_json_path}'")

    val_improved = best_val_loss < initial_val_loss
    improvement_pct = ((initial_val_loss - best_val_loss) / initial_val_loss) * 100.0

    print("\n" + "=" * 78)
    print("      STAGE 8+ EXPERIMENT RESULTS & CHECKPOINTS")
    print("=" * 78)
    print(f"1. Tokenizer Type:          {tokenizer_type.upper()} ({vocab_size} tokens, Leading-Space Pre-tokenization)")
    print(f"2. Dataset Size:            {len(text):,} characters -> {total_tokens:,} tokens (Train: {len(train_tokens):,}, Val: {len(val_tokens):,})")
    print(f"3. Model Parameters:        {total_params:,} (d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}, ctx={context_length})")
    print(f"4. Total Epochs Completed:  {len(history)}")
    print(f"5. Lowest Val Loss:         {best_val_loss:.4f} (Perplexity: {np.exp(best_val_loss):.2f}) at Epoch {best_epoch}")
    print(f"6. Checkpoints Directory:   '{checkpoint_dir}'")
    print(f"   - Checkpoints saved:     {[h['checkpoint'] for h in history]}")
    print(f"   - Best Val Checkpoint:   '{os.path.join(checkpoint_dir, 'checkpoint_best_val.npz')}'")
    if checkpoint_save_path:
        # Only overwrite target checkpoint if explicitly requested
        if best_weights_snapshot is not None:
            restore_best_weights(model, best_weights_snapshot)
        model.save_checkpoint(checkpoint_save_path)
        print(f"   - Explicit target saved: '{checkpoint_save_path}'")
    print("=" * 78 + "\n")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train Scaled Tiny GPT with BPE Subword Tokenization.")
    parser.add_argument("--tokenizer", type=str, default="bpe", choices=["bpe", "char"], help="Tokenizer type: 'bpe' or 'char'")
    parser.add_argument("--num-merges", type=int, default=180, help="Number of BPE merges (default: 180)")
    parser.add_argument("--dataset", type=str, default="data/clean_corpus.txt", help="Path to cleaned dataset text")
    parser.add_argument("--vocab", type=str, default=None, help="Path to save tokenizer vocab JSON")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to save model checkpoint .npz")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save per-epoch checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Maximum epochs to train (default: 5)")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (default: 0 = run all epochs)")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate (default: 0.003)")
    parser.add_argument("--decay", type=float, default=0.01, help="L2 weight decay factor (default: 0.01)")
    parser.add_argument("--context-length", type=int, default=32, help="Context length in tokens (default: 32)")
    parser.add_argument("--d-model", type=int, default=64, help="Embedding dimension (default: 64)")
    parser.add_argument("--n-heads", type=int, default=4, help="Attention heads (default: 4)")
    parser.add_argument("--d-ff", type=int, default=128, help="FFN dimension (default: 128)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--step", type=int, default=2, help="Sequence stride step (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        vocab_save_path=args.vocab,
        checkpoint_save_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_type=args.tokenizer,
        num_merges=args.num_merges,
        max_epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr,
        weight_decay=args.decay,
        context_length=args.context_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        batch_size=args.batch_size,
        step=args.step,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
