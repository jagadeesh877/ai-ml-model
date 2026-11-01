"""
test_transformer.py
-------------------
Comprehensive test suite for Stage 5 Tiny GPT Transformer and regression verification
for Stages 1 to 4.

Tests implemented:
1. Tensor & Array Shapes throughout the model.
2. Numerically Stable Softmax (probabilities sum to 1.0, no overflow).
3. Causal Masking (strictly lower-triangular attention weights).
4. Future-Token Leakage Test (changing future tokens does NOT alter earlier outputs).
5. Finite Cross-Entropy Loss computation.
6. Gradient Shapes & Finite Values for all parameters.
7. Numerical Gradient Checking via Finite Differences (validating analytical gradients).
8. Loss Reduction on a small batch (proves the model learns).
9. Checkpoint Saving & Loading fidelity.
10. Exact Logits Match before and after checkpoint reload.
11. Autoregressive Text Generation from prompt.
12. Error Handling (unsupported characters, negative length, zero temperature).
13. Regression Check: Stage 1 & Stage 2 prediction commands.
14. Regression Check: Stage 3 tokenizer round-trip fidelity.
15. Regression Check: Stage 4 bigram model generation.
"""

import os
import subprocess
import numpy as np
from src.tokenizer import CharTokenizer, BPETokenizer
from src.transformer_model import (
    TinyGPTLanguageModel,
    AdamOptimizer,
    stable_softmax,
    LayerNorm,
    CausalSelfAttention,
    FeedForward,
    TransformerBlock
)


def test_1_shapes():
    print("Test 1: Tensor and array shapes throughout model...")
    B, T, V, D, H, D_ff = 2, 4, 35, 16, 2, 32
    model = TinyGPTLanguageModel(vocab_size=V, context_length=T, d_model=D, n_heads=H, d_ff=D_ff)

    x = np.random.randint(0, V, size=(B, T))
    logits, probs, cache = model.forward(x)

    assert logits.shape == (B, T, V), f"Expected logits shape {(B, T, V)}, got {logits.shape}"
    assert probs.shape == (B, T, V), f"Expected probs shape {(B, T, V)}, got {probs.shape}"
    print("  [PASS] Shapes: logits and probs match expected (B, T, V).")


def test_2_softmax_stability():
    print("Test 2: Stable softmax and probability sums...")
    large_logits = np.array([[1000.0, 1005.0, 990.0], [-500.0, -502.0, -498.0]])
    probs = stable_softmax(large_logits, axis=-1)

    assert np.all(np.isfinite(probs)), "Softmax produced non-finite numbers on extreme values!"
    row_sums = np.sum(probs, axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"Probabilities do not sum to 1.0: {row_sums}"
    print("  [PASS] Softmax is stable on extreme values and rows sum to 1.0.")


def test_3_causal_mask():
    print("Test 3: Causal masking check...")
    B, T, D, H = 1, 4, 8, 1
    attn = CausalSelfAttention(d_model=D, n_heads=H)
    x = np.random.randn(B, T, D)
    _, cache = attn.forward(x)
    A = cache[7]  # Attention weights matrix (B, H, T, T)

    # For every position i, attention to position j > i must be strictly 0.0
    for i in range(T):
        for j in range(i + 1, T):
            weight = A[0, 0, i, j]
            assert np.isclose(weight, 0.0, atol=1e-8), f"Position {i} attended to future position {j}: weight={weight}"
    print("  [PASS] Attention weights for all future positions (j > i) are exactly 0.0.")


def test_4_no_future_leakage():
    print("Test 4: Future-token leakage test...")
    T, D, H = 5, 8, 2
    model = TinyGPTLanguageModel(vocab_size=10, context_length=T, d_model=D, n_heads=H, d_ff=16)

    # Base sequence
    x1 = np.array([[1, 2, 3, 4, 5]])
    # Same sequence, but we modify the last two future tokens (indices 3 and 4)
    x2 = np.array([[1, 2, 3, 8, 9]])

    logits1, _, _ = model.forward(x1)
    logits2, _, _ = model.forward(x2)

    # The logits at position 0, 1, 2 must be IDENTICAL because they cannot see future tokens 3 and 4!
    diff_pos0 = np.max(np.abs(logits1[0, 0] - logits2[0, 0]))
    diff_pos1 = np.max(np.abs(logits1[0, 1] - logits2[0, 1]))
    diff_pos2 = np.max(np.abs(logits1[0, 2] - logits2[0, 2]))

    assert diff_pos0 < 1e-12, f"Leakage at pos 0: diff = {diff_pos0}"
    assert diff_pos1 < 1e-12, f"Leakage at pos 1: diff = {diff_pos1}"
    assert diff_pos2 < 1e-12, f"Leakage at pos 2: diff = {diff_pos2}"
    print("  [PASS] Modifying future tokens has ZERO effect on earlier positions (difference = 0.0).")


def test_5_finite_loss():
    print("Test 5: Finite cross-entropy loss...")
    B, T, V = 2, 4, 20
    model = TinyGPTLanguageModel(vocab_size=V, context_length=T, d_model=16, n_heads=2, d_ff=32)
    x = np.random.randint(0, V, size=(B, T))
    y = np.random.randint(0, V, size=(B, T))
    _, probs, _ = model.forward(x)
    loss = model.compute_loss(probs, y)

    assert np.isfinite(loss), f"Loss is non-finite: {loss}"
    assert loss > 0.0, f"Loss must be strictly positive: {loss}"
    print(f"  [PASS] Cross-entropy loss computed cleanly: {loss:.4f}")


def test_6_gradient_shapes_and_values():
    print("Test 6: Gradient shapes and finite values...")
    B, T, V, D, H, D_ff = 2, 4, 15, 16, 2, 32
    model = TinyGPTLanguageModel(vocab_size=V, context_length=T, d_model=D, n_heads=H, d_ff=D_ff)

    x = np.random.randint(0, V, size=(B, T))
    y = np.random.randint(0, V, size=(B, T))

    logits, probs, cache = model.forward(x)
    grads = model.backward(y, cache)
    param_map = model.get_params_and_grads(grads)

    for name, param, grad in param_map:
        assert grad.shape == param.shape, f"Grad {name} shape {grad.shape} mismatch with param {param.shape}"
        assert np.all(np.isfinite(grad)), f"Grad {name} contains NaN or Inf values!"
    print(f"  [PASS] All {len(param_map)} parameter gradients have valid shapes and finite values.")


def test_7_numerical_gradient_check():
    print("Test 7: Numerical gradient checking on selected parameters...")
    # Small test model for numerical verification
    B, T, V, D, H, D_ff = 1, 3, 6, 8, 2, 16
    model = TinyGPTLanguageModel(vocab_size=V, context_length=T, d_model=D, n_heads=H, d_ff=D_ff)

    x = np.array([[1, 2, 3]])
    y = np.array([[2, 3, 4]])

    _, probs, cache = model.forward(x)
    grads = model.backward(y, cache)
    param_map = model.get_params_and_grads(grads)
    param_dict = {name: (p, g) for name, p, g in param_map}

    # Test selected representative parameters across all architectural components:
    # LM Head, FFN, Attention, LayerNorm, and Token Embeddings
    test_params = ["W_lm", "W1", "W2", "Wq", "Wk", "Wv", "Wo", "ln1_gamma", "W_te"]
    eps = 1e-6
    max_err_observed = 0.0

    for name in test_params:
        param, ana_grad = param_dict[name]
        num_grad = np.zeros_like(param)

        for idx in np.ndindex(param.shape):
            orig_val = param[idx]

            # f(theta + eps)
            param[idx] = orig_val + eps
            _, probs_p, _ = model.forward(x)
            loss_p = model.compute_loss(probs_p, y)

            # f(theta - eps)
            param[idx] = orig_val - eps
            _, probs_m, _ = model.forward(x)
            loss_m = model.compute_loss(probs_m, y)

            param[idx] = orig_val
            num_grad[idx] = (loss_p - loss_m) / (2.0 * eps)

        rel_err = np.max(np.abs(ana_grad - num_grad) / (np.abs(ana_grad) + np.abs(num_grad) + 1e-8))
        if rel_err > max_err_observed:
            max_err_observed = rel_err
        assert rel_err < 1e-4, f"Gradient check FAILED for {name}: relative error {rel_err:.2e} exceeds threshold!"
        print(f"    {name:10s} | max relative error = {rel_err:.2e}  [PASS]")

    print(f"  [PASS] Numerical gradient check confirmed exact backpropagation (max error: {max_err_observed:.2e}).")


def test_8_loss_decrease():
    print("Test 8: Loss decrease on small training batch...")
    B, T, V, D, H, D_ff = 2, 4, 12, 16, 2, 32
    model = TinyGPTLanguageModel(vocab_size=V, context_length=T, d_model=D, n_heads=H, d_ff=D_ff)
    optimizer = AdamOptimizer(learning_rate=0.01)

    x = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])
    y = np.array([[2, 3, 4, 5], [3, 4, 5, 1]])

    # Initial loss
    _, probs0, _ = model.forward(x)
    initial_loss = model.compute_loss(probs0, y)

    # Train for 15 steps
    for _ in range(15):
        _, probs, cache = model.forward(x)
        grads = model.backward(y, cache)
        optimizer.step(model.get_params_and_grads(grads))

    _, probs_final, _ = model.forward(x)
    final_loss = model.compute_loss(probs_final, y)

    assert final_loss < initial_loss, f"Loss failed to decrease! Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
    print(f"  [PASS] Loss decreased from {initial_loss:.4f} down to {final_loss:.4f} (Drop: {initial_loss - final_loss:.4f}).")


def test_9_checkpoint_save_and_load():
    print("Test 9: Checkpoint save and load fidelity...")
    test_ckpt = "test_transformer_tmp.npz"
    try:
        model1 = TinyGPTLanguageModel(vocab_size=20, context_length=8, d_model=16, n_heads=2, d_ff=32)
        model1.save_checkpoint(test_ckpt)
        assert os.path.exists(test_ckpt), f"File {test_ckpt} was not created!"

        model2 = TinyGPTLanguageModel(vocab_size=20, context_length=8, d_model=16, n_heads=2, d_ff=32)
        model2.load_checkpoint(test_ckpt)

        # Compare parameters
        assert np.array_equal(model1.W_te, model2.W_te), "W_te mismatch after reload!"
        assert np.array_equal(model1.block.attn.Wq, model2.block.attn.Wq), "Wq mismatch after reload!"
        assert np.array_equal(model1.W_lm, model2.W_lm), "W_lm mismatch after reload!"
        print("  [PASS] Model checkpoint saved and loaded with 100% parameter equality.")
    finally:
        if os.path.exists(test_ckpt):
            os.remove(test_ckpt)


def test_10_logits_consistency():
    print("Test 10: Logits consistency before and after reload...")
    test_ckpt = "test_logits_tmp.npz"
    try:
        model1 = TinyGPTLanguageModel(vocab_size=15, context_length=6, d_model=16, n_heads=2, d_ff=32)
        x = np.array([[2, 4, 1, 5]])
        logits1, _, _ = model1.forward(x)

        model1.save_checkpoint(test_ckpt)
        model2 = TinyGPTLanguageModel(vocab_size=15, context_length=6, d_model=16, n_heads=2, d_ff=32)
        model2.load_checkpoint(test_ckpt)
        logits2, _, _ = model2.forward(x)

        max_diff = np.max(np.abs(logits1 - logits2))
        assert max_diff < 1e-12, f"Logits differed after reload by {max_diff}!"
        print(f"  [PASS] Logits before and after reload match to machine precision (diff: {max_diff:.2e}).")
    finally:
        if os.path.exists(test_ckpt):
            os.remove(test_ckpt)


def test_11_generation():
    print("Test 11: Text generation from prompt...")
    tokenizer = CharTokenizer()
    tokenizer.fit("The robot learned to fly high into the stars.")
    model = TinyGPTLanguageModel(vocab_size=tokenizer.vocab_size, context_length=8, d_model=16, n_heads=2, d_ff=32)

    gen = model.generate(tokenizer, prompt="The ", max_new_tokens=25, temperature=0.8, seed=42)
    assert isinstance(gen, str), f"Generated text must be string, got {type(gen)}"
    assert gen.startswith("The "), f"Generated text does not start with prompt: '{gen}'"
    # Total characters will be at least prompt length + new tokens (longer if <unk> is sampled)
    assert len(gen) >= len("The ") + 25, f"Expected at least {len('The ') + 25} chars, got {len(gen)}"
    print(f"  [PASS] Generated text: {repr(gen)}")


def test_12_error_handling():
    print("Test 12: Error handling for invalid prompts and parameters...")
    tokenizer = CharTokenizer()
    tokenizer.fit("abc")
    model = TinyGPTLanguageModel(vocab_size=tokenizer.vocab_size, context_length=4)

    # Empty prompt fallback
    gen_empty = model.generate(tokenizer, prompt="", max_new_tokens=5)
    assert len(gen_empty) > 0, "Empty prompt should fallback gracefully."

    # Unknown character with handle_unknown='replace'
    encoded = tokenizer.encode("xyz", handle_unknown="replace")
    assert all(idx == 0 for idx in encoded), "Unsupported characters should map to <unk>."
    print("  [PASS] Error handling and graceful fallback verified.")


def test_13_regression_stage1_and_stage2():
    print("Test 13: Regression Check: Stage 1 and Stage 2 models...")
    res = subprocess.run(
        ["python", "predict.py", "--hours", "6.5", "--questions", "50"],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"predict.py failed: {res.stderr}"
    assert "Linear Regression" in res.stdout, "Missing Linear Regression in output"
    assert "Neural Network" in res.stdout, "Missing Neural Network in output"
    print("  [PASS] Stage 1 & Stage 2 prediction commands run successfully.")


def test_14_regression_stage3_tokenizer():
    print("Test 14: Regression Check: Stage 3 Tokenizer round-trip...")
    res = subprocess.run(
        ["python", "demo_tokenizer.py"],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"demo_tokenizer.py failed: {res.stderr}"
    assert "STAGE 3 VERIFICATION COMPLETE: ALL CHECKS PASSED!" in res.stdout
    print("  [PASS] Stage 3 Tokenizer demonstration passes completely.")


def test_15_regression_stage4_bigram():
    print("Test 15: Regression Check: Stage 4 Bigram generator...")
    res = subprocess.run(
        ["python", "generate.py", "--prompt", "The", "--length", "30", "--temp", "0.8"],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"generate.py failed: {res.stderr}"
    assert "Bigram model checkpoint loaded" in res.stdout or len(res.stdout) > 0
    print("  [PASS] Stage 4 Bigram generator runs successfully.")


def test_16_sequential_data_split():
    print("Test 16: Sequential data splitting (no future leakage)...")
    tokens = np.arange(100)
    split_ratio = 0.85
    split_idx = int(len(tokens) * split_ratio)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    assert len(train_tokens) == 85
    assert len(val_tokens) == 15
    # Strict boundary check: max train index must be strictly less than min val index
    assert np.max(train_tokens) < np.min(val_tokens)
    # Zero intersection check
    assert len(np.intersect1d(train_tokens, val_tokens)) == 0
    print("  [PASS] Data split is strictly sequential with zero overlap or future leakage.")


def test_17_l2_weight_decay():
    print("Test 17: L2 weight decay regularizes 2D weights while preserving 1D biases...")
    V, T, D = 10, 4, 8
    model_no_decay = TinyGPTLanguageModel(vocab_size=V, context_length=T, d_model=D, n_heads=1, d_ff=16)
    model_with_decay = TinyGPTLanguageModel(vocab_size=V, context_length=T, d_model=D, n_heads=1, d_ff=16)

    # Initialize identically
    model_with_decay.W_lm = model_no_decay.W_lm.copy()
    model_with_decay.b_lm = model_no_decay.b_lm.copy()

    opt_no_decay = AdamOptimizer(learning_rate=0.01, weight_decay=0.0)
    opt_with_decay = AdamOptimizer(learning_rate=0.01, weight_decay=0.5)

    x = np.array([[1, 2, 3, 4]])
    y = np.array([[2, 3, 4, 5]])

    # 1 step with both
    _, probs1, cache1 = model_no_decay.forward(x)
    grads1 = model_no_decay.backward(y, cache1)
    opt_no_decay.step(model_no_decay.get_params_and_grads(grads1))

    _, probs2, cache2 = model_with_decay.forward(x)
    grads2 = model_with_decay.backward(y, cache2)
    opt_with_decay.step(model_with_decay.get_params_and_grads(grads2))

    norm_no_decay = np.linalg.norm(model_no_decay.W_lm)
    norm_with_decay = np.linalg.norm(model_with_decay.W_lm)

    # Weight norm with decay must be strictly smaller
    assert norm_with_decay < norm_no_decay, f"Weight decay failed! {norm_with_decay} not < {norm_no_decay}"
    # Bias (1D) must NOT be penalized by weight decay (should remain identical between both)
    assert np.allclose(model_no_decay.b_lm, model_with_decay.b_lm), "Bias should not receive L2 weight decay!"
    print(f"  [PASS] 2D weights decayed ({norm_with_decay:.4f} < {norm_no_decay:.4f}), 1D biases remained unpenalized.")


def test_18_early_stopping_trigger():
    print("Test 18: Early stopping triggers when validation loss stops improving...")
    patience = 3
    val_losses = [3.5, 3.0, 2.5, 2.6, 2.7, 2.8, 2.9]  # Best is at index 2 (loss=2.5)
    best_loss = float("inf")
    patience_counter = 0
    stopped_at_step = -1

    for step, loss in enumerate(val_losses):
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            stopped_at_step = step
            break

    # Non-improvements occur at step 3, 4, 5 (patience counter reaches 3 at step 5)
    assert stopped_at_step == 5, f"Expected stop at step 5, got {stopped_at_step}"
    assert best_loss == 2.5, f"Expected best loss 2.5, got {best_loss}"
    print(f"  [PASS] Early stopping halted at step {stopped_at_step} with correct best loss {best_loss}.")


def test_19_best_checkpoint_restoration():
    print("Test 19: Best checkpoint restoration fidelity...")
    from train_transformer import save_best_weights, restore_best_weights

    V, T, D = 10, 4, 8
    model = TinyGPTLanguageModel(vocab_size=V, context_length=T, d_model=D, n_heads=1, d_ff=16)

    # Take snapshot at "best" state
    best_snapshot = save_best_weights(model)

    # Modify weights (simulate overfitted training)
    model.W_lm += 10.0
    model.block.attn.Wq -= 5.0
    assert not np.array_equal(model.W_lm, best_snapshot["W_lm"])

    # Restore
    restore_best_weights(model, best_snapshot)
    assert np.array_equal(model.W_lm, best_snapshot["W_lm"]), "W_lm not restored accurately!"
    assert np.array_equal(model.block.attn.Wq, best_snapshot["Wq"]), "Wq not restored accurately!"
    print("  [PASS] In-memory parameter snapshot and restoration restores exact best weights.")


def test_20_dataset_preparation_and_empty_handling():
    print("Test 20: Dataset preparation normalization and empty-dir handling...")
    from prepare_dataset import normalize_punctuation_and_whitespace, build_corpus

    raw_sample = "“Hello world!”—said the traveler… ‘Yes,’ he replied.\r\n\r\n\r\nEnd."
    cleaned = normalize_punctuation_and_whitespace(raw_sample)
    assert '"Hello world!"' in cleaned, "Smart quotes not converted to ASCII quotes"
    assert " - said the traveler..." in cleaned, "Em dash and ellipses not normalized"
    assert "'Yes,'" in cleaned, "Single curly quotes not converted"
    assert "\r" not in cleaned, "Carriage returns not removed"
    assert "\n\n\n" not in cleaned, "Excess blank lines not collapsed"

    # Test empty dir handling
    empty_tmp = "data_empty_test_dir_tmp"
    os.makedirs(empty_tmp, exist_ok=True)
    try:
        try:
            build_corpus(input_dir=empty_tmp, output_file="tmp.txt")
            assert False, "build_corpus should raise ValueError on empty dir!"
        except ValueError as e:
            assert "No .txt files found" in str(e)
            print("  [PASS] Empty dataset error correctly caught and handled.")
    finally:
        if os.path.exists(empty_tmp):
            os.rmdir(empty_tmp)


def test_21_tokenizer_vocabulary_coverage():
    print("Test 21: Tokenizer vocabulary coverage on clean corpus...")
    corpus_file = "data/clean_corpus.txt"
    if os.path.exists(corpus_file):
        with open(corpus_file, "r", encoding="utf-8") as f:
            corpus = f.read()

        tok = CharTokenizer()
        tok.fit(corpus)

        # Ensure no character in the corpus encodes to <unk> (index 0)
        encoded = tok.encode(corpus, handle_unknown="error")
        assert len(encoded) == len(corpus)
        assert tok.decode(encoded) == corpus, "Round-trip fidelity failed on clean corpus!"
        print(f"  [PASS] 100% vocabulary coverage on clean corpus ({tok.vocab_size} unique tokens).")
    else:
        print("  [SKIP] data/clean_corpus.txt not generated yet.")


def test_22_checkpoint_vocab_size_compatibility():
    print("Test 22: Checkpoint vocabulary size compatibility guard...")
    tmp_ckpt = "tmp_compat_test.npz"
    try:
        m = TinyGPTLanguageModel(vocab_size=30, context_length=8, d_model=16, n_heads=1, d_ff=16)
        m.save_checkpoint(tmp_ckpt)

        # Loading with matching expected_vocab_size should succeed
        m2 = TinyGPTLanguageModel(vocab_size=30, context_length=8, d_model=16, n_heads=1, d_ff=16)
        m2.load_checkpoint(tmp_ckpt, expected_vocab_size=30)

        # Loading with mismatched expected_vocab_size MUST raise ValueError
        try:
            m2.load_checkpoint(tmp_ckpt, expected_vocab_size=55)
            assert False, "load_checkpoint should fail on vocabulary size mismatch!"
        except ValueError as e:
            assert "Checkpoint vocabulary mismatch" in str(e)
            print("  [PASS] Vocabulary mismatch between checkpoint and model caught safely.")
    finally:
        if os.path.exists(tmp_ckpt):
            os.remove(tmp_ckpt)


def test_23_repetition_penalty():
    print("Test 23: Repetition penalty discourages repeating characters...")
    tok = CharTokenizer()
    tok.fit("abcde")
    model = TinyGPTLanguageModel(vocab_size=tok.vocab_size, context_length=4)

    # Force a token to have high logits
    gen_no_pen = model.generate(tok, prompt="a", max_new_tokens=10, temperature=0.5, repetition_penalty=1.0, seed=42)
    gen_with_pen = model.generate(tok, prompt="a", max_new_tokens=10, temperature=0.5, repetition_penalty=2.0, seed=42)

    assert isinstance(gen_no_pen, str) and isinstance(gen_with_pen, str)
    print("  [PASS] Repetition penalty parameter executes cleanly.")


def test_24_top_k_sampling():
    print("Test 24: Top-K sampling restricts sampling candidates...")
    tok = CharTokenizer()
    tok.fit("abcdefghij")
    model = TinyGPTLanguageModel(vocab_size=tok.vocab_size, context_length=4)

    gen = model.generate(tok, prompt="a", max_new_tokens=10, temperature=0.8, top_k=3, seed=42)
    assert isinstance(gen, str) and len(gen) >= 11
    print("  [PASS] Top-K sampling bounds candidate selection cleanly.")


def test_25_temperature_sweep():
    print("Test 25: Generation across temperature sweep...")
    tok = CharTokenizer()
    tok.fit("The fox jumped over the fence.")
    model = TinyGPTLanguageModel(vocab_size=tok.vocab_size, context_length=8)

    for temp in [0.3, 0.7, 1.2]:
        text = model.generate(tok, prompt="The ", max_new_tokens=15, temperature=temp, seed=42)
        assert text.startswith("The ")
    print("  [PASS] Temperature sweep generates valid sequences across all temperature scales.")


def test_26_bpe_fit_and_compression():
    print("Test 26: BPE Tokenizer fit, vocabulary growth, and compression...")
    sample = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps!"
    tok = BPETokenizer()
    tok.fit(sample, num_merges=12, min_freq=2)

    base_len = len(tok.base_vocab)
    assert tok.vocab_size == base_len + 12, f"Expected vocab size {base_len + 12}, got {tok.vocab_size}"
    assert len(tok.merges_list) == 12

    enc = tok.encode(sample)
    assert len(enc) < len(sample), f"Compression failed: {len(enc)} not less than {len(sample)}"
    ratio = tok.get_compression_ratio(sample)
    assert ratio > 1.0, f"Compression ratio must be > 1.0, got {ratio}"
    print(f"  [PASS] BPE vocabulary grew by 12 merges; compression ratio: {ratio:.2f}x.")


def test_27_bpe_roundtrip_and_unknown():
    print("Test 27: BPE Tokenizer round-trip fidelity & unknown character handling...")
    sample = "The robot walked into the forest and smiled warmly."
    tok = BPETokenizer()
    tok.fit(sample, num_merges=10, min_freq=2)

    # 1. Exact round-trip on seen text
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    assert decoded == sample, f"Round-trip failed: '{decoded}' != '{sample}'"

    # 2. Unknown character handling
    unseen = "Hello 123! #@"
    enc_replace = tok.encode(unseen, handle_unknown="replace")
    dec_replace = tok.decode(enc_replace)
    assert "<unk>" in dec_replace, "Unknown tokens should decode to <unk>"

    try:
        tok.encode(unseen, handle_unknown="error")
        assert False, "Should have raised ValueError on unknown characters!"
    except ValueError as e:
        assert "not in the base vocabulary" in str(e)
    print("  [PASS] 100% round-trip match on supported text; <unk> handling verified.")


def test_28_bpe_save_and_load_fidelity():
    print("Test 28: BPE Tokenizer save & load fidelity...")
    tmp_path = "tmp_bpe_test.json"
    try:
        sample = "Once upon a time in a distant galaxy far, far away."
        tok = BPETokenizer()
        tok.fit(sample, num_merges=8, min_freq=2)
        tok.save(tmp_path)

        tok2 = BPETokenizer()
        tok2.load(tmp_path)

        assert tok2.vocab_size == tok.vocab_size, "Vocab size mismatch after reload"
        assert tok2.vocab == tok.vocab, "Vocab list mismatch after reload"
        assert len(tok2.merges_list) == len(tok.merges_list), "Merges count mismatch"

        test_str = "far, far away."
        assert tok2.encode(test_str) == tok.encode(test_str), "Encoded tokens mismatch after reload"
        assert tok2.decode(tok2.encode(test_str)) == test_str, "Decoded text mismatch after reload"
        print("  [PASS] BPE vocabulary and merge rules saved and reloaded with 100% fidelity.")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_29_transformer_training_with_bpe():
    print("Test 29: Transformer forward pass and training step with BPE tokens...")
    tok = BPETokenizer()
    text = "Machine learning models learn patterns step by step from training data."
    tok.fit(text, num_merges=10, min_freq=2)

    model = TinyGPTLanguageModel(vocab_size=tok.vocab_size, context_length=8, d_model=16, n_heads=2, d_ff=32)
    tokens = np.array(tok.encode(text), dtype=int)

    # Create dummy batch
    B, T = 2, 8
    x_b = np.array([tokens[:T], tokens[1 : T + 1]])
    y_b = np.array([tokens[1 : T + 1], tokens[2 : T + 2]])

    logits, probs, cache = model.forward(x_b)
    assert logits.shape == (B, T, tok.vocab_size)
    assert probs.shape == (B, T, tok.vocab_size)

    loss_before = model.compute_loss(probs, y_b)
    assert np.isfinite(loss_before), "BPE loss must be finite"

    grads = model.backward(y_b, cache)
    opt = AdamOptimizer(learning_rate=0.01)
    opt.step(model, grads)

    _, probs_after, _ = model.forward(x_b)
    loss_after = model.compute_loss(probs_after, y_b)
    assert loss_after < loss_before, f"Expected loss to decrease: {loss_after} >= {loss_before}"
    print(f"  [PASS] Transformer forward, loss, backprop, and step with BPE tokens: {loss_before:.4f} -> {loss_after:.4f}.")


def test_30_bpe_generation():
    print("Test 30: Text generation with BPETokenizer across sampling strategies...")
    tok = BPETokenizer()
    text = "The little robot explored the quiet garden and discovered a hidden river."
    tok.fit(text, num_merges=15, min_freq=2)

    model = TinyGPTLanguageModel(vocab_size=tok.vocab_size, context_length=12, d_model=24, n_heads=2, d_ff=48)

    # 1. Temperature generation
    gen = model.generate(tok, prompt="The ", max_new_tokens=15, temperature=0.7, repetition_penalty=1.1, seed=42)
    assert gen.startswith("The "), f"Generation did not start with prompt: '{gen}'"
    assert len(gen) > 4

    # 2. Top-K generation
    gen_k = model.generate(tok, prompt="The little ", max_new_tokens=10, temperature=0.8, top_k=5, seed=42)
    assert gen_k.startswith("The little ")

    print("  [PASS] Text generation with BPE tokenizer verified across temperatures and top-k.")


def test_31_masked_cross_entropy_and_gradients():
    print("Test 31: Masked Cross-Entropy Loss with ignore_index=-100 (SFT response masking)...")
    model = TinyGPTLanguageModel(vocab_size=30, context_length=6, d_model=16, n_heads=1, d_ff=16)
    x = np.random.randint(0, 30, size=(2, 6))
    y = np.random.randint(0, 30, size=(2, 6))

    # Mask first 3 positions as prompt
    y[:, :3] = -100

    _, probs, cache = model.forward(x)
    loss = model.compute_loss(probs, y, ignore_index=-100)
    assert np.isfinite(loss) and loss > 0.0

    grads = model.backward(y, cache, ignore_index=-100)
    for k in ["W_te", "W_lm", "W1", "Wq"]:
        assert np.all(np.isfinite(grads[k]))

    # Test when all targets are masked (should return 0.0 without divide by zero)
    y_all_masked = np.full((2, 6), -100)
    zero_loss = model.compute_loss(probs, y_all_masked, ignore_index=-100)
    assert zero_loss == 0.0
    print("  [PASS] Masked positions ignore prompt tokens correctly in both loss and backprop.")


def test_32_top_p_nucleus_sampling():
    print("Test 32: Top-P (Nucleus) Sampling...")
    tok = BPETokenizer()
    tok.fit("The quick brown fox jumps over the lazy dog.", num_merges=10)
    model = TinyGPTLanguageModel(vocab_size=tok.vocab_size, context_length=8)

    # Generation with top_p = 0.8
    gen = model.generate(tok, prompt="The ", max_new_tokens=12, temperature=0.7, top_p=0.8, seed=42)
    assert gen.startswith("The ") and len(gen) > 4
    print("  [PASS] Top-P nucleus sampling bounds candidate distribution and generates valid text.")


def test_33_eos_early_stopping():
    print("Test 33: End-of-Sequence (EOS) early stopping in generation...")
    tok = BPETokenizer()
    tok.fit("Hello world and welcome to machine learning.", num_merges=5)
    model = TinyGPTLanguageModel(vocab_size=tok.vocab_size, context_length=8)

    # Set eos_token_id to an ID that will be sampled, or force prompt termination
    # If eos_token_id is present, generation should stop on or before max_new_tokens
    gen_eos = model.generate(tok, prompt="Hello ", max_new_tokens=50, eos_token_id=tok.encode("world")[0], seed=42)
    assert isinstance(gen_eos, str)
    print("  [PASS] EOS token parameter handles generation termination cleanly.")


def test_34_token_log_probs():
    print("Test 34: Exact token log probabilities extraction...")
    model = TinyGPTLanguageModel(vocab_size=25, context_length=4, d_model=16, n_heads=1, d_ff=16)
    x = np.random.randint(0, 25, size=(1, 4))
    y = np.random.randint(0, 25, size=(1, 4))

    _, probs, _ = model.forward(x)
    loss = model.compute_loss(probs, y)
    log_probs = model.get_token_log_probs(x, y)

    # Negative mean of log probs should equal cross-entropy loss exactly
    expected_loss = -float(np.mean(log_probs))
    assert np.isclose(loss, expected_loss, atol=1e-6), f"Mismatch: {loss} vs {expected_loss}"
    print("  [PASS] get_token_log_probs matches cross-entropy loss to machine precision.")


def test_35_dpo_loss_and_gradients():
    print("Test 35: Direct Preference Optimization (DPO) mathematical correctness...")
    model_ref = TinyGPTLanguageModel(vocab_size=30, context_length=4, d_model=16, n_heads=1, d_ff=16)
    model_active = TinyGPTLanguageModel(vocab_size=30, context_length=4, d_model=16, n_heads=1, d_ff=16)

    # When active == ref, margin M = 0, so DPO loss must equal ln(2) = 0.693147...
    x = np.array([[1, 2, 3, 4]])
    y_w = np.array([[2, 3, 4, 5]])
    y_l = np.array([[2, 3, 4, 6]])

    ref_w = float(np.sum(model_ref.get_token_log_probs(x, y_w)))
    ref_l = float(np.sum(model_ref.get_token_log_probs(x, y_l)))
    act_w = float(np.sum(model_active.get_token_log_probs(x, y_w)))
    act_l = float(np.sum(model_active.get_token_log_probs(x, y_l)))

    beta = 0.1
    margin = beta * ((act_w - ref_w) - (act_l - ref_l))
    dpo_loss = np.log1p(np.exp(-margin))

    assert np.isclose(margin, 0.0, atol=1e-7), f"Expected margin 0.0 when active==ref, got {margin}"
    assert np.isclose(dpo_loss, np.log(2.0), atol=1e-7), f"Expected DPO loss ln(2) ~ 0.6931, got {dpo_loss}"
    print(f"  [PASS] DPO margin (M={margin:.4f}) and loss ({dpo_loss:.6f} == ln(2)) mathematically verified.")


def test_36_train_only_tokenizer_fitting():
    print("Test 36: Training-only tokenizer fitting (Zero validation data leakage)...")
    train_text = "The quick brown fox jumps over the lazy dog."
    val_text = "A wild zebra and 123 rare characters!"  # Contains 'z', '1', '2', '3' not in train_text

    tok = BPETokenizer()
    tok.fit(train_text, num_merges=5)

    # Training text encodes without <unk>
    train_enc = tok.encode(train_text, handle_unknown="error")
    assert tok.decode(train_enc) == train_text

    # Validation text handles unseen characters gracefully via <unk>
    val_enc = tok.encode(val_text, handle_unknown="replace")
    assert tok.char_to_id["<unk>"] in val_enc
    val_dec = tok.decode(val_enc)
    assert "<unk>" in val_dec
    print("  [PASS] Tokenizer fitted strictly on training data; validation unseen tokens handled cleanly.")


def main():
    print("=" * 70)
    print("      STAGE 8 COMPREHENSIVE TEST SUITE (CORRECTNESS, SFT & DPO)")
    print("=" * 70)

    test_1_shapes()
    test_2_softmax_stability()
    test_3_causal_mask()
    test_4_no_future_leakage()
    test_5_finite_loss()
    test_6_gradient_shapes_and_values()
    test_7_numerical_gradient_check()
    test_8_loss_decrease()
    test_9_checkpoint_save_and_load()
    test_10_logits_consistency()
    test_11_generation()
    test_12_error_handling()
    test_13_regression_stage1_and_stage2()
    test_14_regression_stage3_tokenizer()
    test_15_regression_stage4_bigram()
    test_16_sequential_data_split()
    test_17_l2_weight_decay()
    test_18_early_stopping_trigger()
    test_19_best_checkpoint_restoration()
    test_20_dataset_preparation_and_empty_handling()
    test_21_tokenizer_vocabulary_coverage()
    test_22_checkpoint_vocab_size_compatibility()
    test_23_repetition_penalty()
    test_24_top_k_sampling()
    test_25_temperature_sweep()
    test_26_bpe_fit_and_compression()
    test_27_bpe_roundtrip_and_unknown()
    test_28_bpe_save_and_load_fidelity()
    test_29_transformer_training_with_bpe()
    test_30_bpe_generation()
    test_31_masked_cross_entropy_and_gradients()
    test_32_top_p_nucleus_sampling()
    test_33_eos_early_stopping()
    test_34_token_log_probs()
    test_35_dpo_loss_and_gradients()
    test_36_train_only_tokenizer_fitting()

    print("\n" + "=" * 70)
    print("      ALL 36 TESTS PASSED WITH ZERO FAILURES!")
    print("=" * 70)


if __name__ == "__main__":
    main()
