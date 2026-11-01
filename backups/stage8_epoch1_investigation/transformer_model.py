"""
transformer_model.py
--------------------
Stage 5: Tiny GPT-Style Transformer from Scratch (Pure Python & NumPy).

Why does GPT use a Transformer instead of a Bigram model?
- A Bigram model only remembers 1 past character. It cannot remember what came 3 letters ago.
- A Transformer uses Causal Self-Attention: every character can look back at ALL previous
  characters in its context window to decide what character comes next!

Core Architecture:
1. Token Embeddings: Turns integer token IDs into rich continuous feature vectors.
2. Positional Embeddings: Adds vectors representing position (0, 1, 2, ...) so the model
   knows the order of characters.
3. Layer Normalization: Keeps activations well-scaled for stable gradient flow.
4. Causal Masked Self-Attention: Lets tokens exchange information with past tokens only.
5. Feed-Forward Network (FFN): Processes each token's representation individually with non-linearity.
6. Residual Connections: Skips connections (x + f(x)) that prevent vanishing gradients.
7. Language Model Output Head: Projects hidden vectors to vocabulary logits for next-token prediction.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable Softmax activation.
    Subtracts max(x) along the axis to prevent floating-point overflow (e.g. exp(1000) -> inf).
    """
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class LayerNorm:
    """
    Layer Normalization:
    Normalizes features across the embedding dimension D for each token independently.
    
    Formula:
        y = gamma * ((x - mean) / sqrt(var + eps)) + beta
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        # Trainable scale (gamma) initialized to 1, bias (beta) initialized to 0
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """
        x shape: (B, T, D)
        Returns: normalized x and cache for backpropagation
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - mean) * std_inv
        out = self.gamma * x_hat + self.beta
        cache = (x, x_hat, mean, var, std_inv)
        return out, cache

    def backward(self, dout: np.ndarray, cache: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Analytical backpropagation through Layer Normalization.
        """
        x, x_hat, mean, var, std_inv = cache
        dgamma = np.sum(dout * x_hat, axis=(0, 1))
        dbeta = np.sum(dout, axis=(0, 1))
        dx_hat = dout * self.gamma
        
        # Exact gradient w.r.t input x
        dx = std_inv * (
            dx_hat - np.mean(dx_hat, axis=-1, keepdims=True)
            - x_hat * np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
        )
        return dx, dgamma, dbeta


class CausalSelfAttention:
    """
    Causal (Masked) Multi-Head Self-Attention:
    Allows each token to attend to all preceding tokens (and itself), but NEVER future tokens.
    
    Formula:
        Attention(Q, K, V) = softmax((Q @ K.T) / sqrt(d_k) + causal_mask) @ V
    """
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Weight matrices for Query, Key, Value projections and Output projection
        scale = 1.0 / np.sqrt(d_model)
        self.Wq = np.random.randn(d_model, d_model) * scale
        self.bq = np.zeros(d_model)

        self.Wk = np.random.randn(d_model, d_model) * scale
        self.bk = np.zeros(d_model)

        self.Wv = np.random.randn(d_model, d_model) * scale
        self.bv = np.zeros(d_model)

        self.Wo = np.random.randn(d_model, d_model) * scale
        self.bo = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """
        Forward Pass:
        x shape: (B, T, D)
        """
        B, T, D = x.shape
        H = self.n_heads
        d_h = self.d_head

        # 1. Project to Q, K, V
        Q = np.dot(x, self.Wq) + self.bq
        K = np.dot(x, self.Wk) + self.bk
        V = np.dot(x, self.Wv) + self.bv

        # 2. Reshape into multiple heads: (B, H, T, d_head)
        Q_h = Q.reshape(B, T, H, d_h).transpose(0, 2, 1, 3)
        K_h = K.reshape(B, T, H, d_h).transpose(0, 2, 1, 3)
        V_h = V.reshape(B, T, H, d_h).transpose(0, 2, 1, 3)

        # 3. Scaled Dot-Product: (B, H, T, T)
        scale = 1.0 / np.sqrt(d_h)
        scores = np.matmul(Q_h, K_h.transpose(0, 1, 3, 2)) * scale

        # 4. Causal Mask: Set future positions (i < j) to -1e9 so their softmax weight is 0.0
        mask = np.triu(np.ones((T, T)), k=1).astype(bool)
        scores[:, :, mask] = -1e9

        # 5. Softmax attention weights
        A = stable_softmax(scores, axis=-1)

        # 6. Weighted values: (B, H, T, d_head)
        out_h = np.matmul(A, V_h)

        # 7. Concatenate heads back to (B, T, D)
        out_cat = out_h.transpose(0, 2, 1, 3).reshape(B, T, D)

        # 8. Output projection
        out = np.dot(out_cat, self.Wo) + self.bo

        cache = (x, Q, K, V, Q_h, K_h, V_h, A, out_cat, scale)
        return out, cache

    def backward(self, dout: np.ndarray, cache: Tuple) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Backpropagation through Multi-Head Causal Attention.
        """
        x, Q, K, V, Q_h, K_h, V_h, A, out_cat, scale = cache
        B, T, D = x.shape
        H = self.n_heads
        d_h = self.d_head

        # Gradients for Output Projection
        dWo = np.tensordot(out_cat, dout, axes=([0, 1], [0, 1]))
        dbo = np.sum(dout, axis=(0, 1))
        dout_cat = np.dot(dout, self.Wo.T)

        # Reshape to multi-head
        dout_h = dout_cat.reshape(B, T, H, d_h).transpose(0, 2, 1, 3)

        # out_h = A @ V_h
        dA = np.matmul(dout_h, V_h.transpose(0, 1, 3, 2))
        dV_h = np.matmul(A.transpose(0, 1, 3, 2), dout_h)

        # Softmax backward: dS = A * (dA - sum(dA * A)) * scale
        dS = A * (dA - np.sum(dA * A, axis=-1, keepdims=True)) * scale

        # Q and K gradients
        dQ_h = np.matmul(dS, K_h)
        dK_h = np.matmul(dS.transpose(0, 1, 3, 2), Q_h)

        # Reshape back to (B, T, D)
        dQ = dQ_h.transpose(0, 2, 1, 3).reshape(B, T, D)
        dK = dK_h.transpose(0, 2, 1, 3).reshape(B, T, D)
        dV = dV_h.transpose(0, 2, 1, 3).reshape(B, T, D)

        # Projection weights gradients
        dWq = np.tensordot(x, dQ, axes=([0, 1], [0, 1]))
        dbq = np.sum(dQ, axis=(0, 1))

        dWk = np.tensordot(x, dK, axes=([0, 1], [0, 1]))
        dbk = np.sum(dK, axis=(0, 1))

        dWv = np.tensordot(x, dV, axes=([0, 1], [0, 1]))
        dbv = np.sum(dV, axis=(0, 1))

        # Gradient w.r.t input x
        dx = np.dot(dQ, self.Wq.T) + np.dot(dK, self.Wk.T) + np.dot(dV, self.Wv.T)

        grads = {
            "Wq": dWq, "bq": dbq,
            "Wk": dWk, "bk": dbk,
            "Wv": dWv, "bv": dbv,
            "Wo": dWo, "bo": dbo
        }
        return dx, grads


class FeedForward:
    """
    Position-Wise Feed-Forward Network:
    Two linear transformations with a non-linear ReLU activation in between.
    
    Formula:
        FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
    """
    def __init__(self, d_model: int, d_ff: int):
        scale1 = 1.0 / np.sqrt(d_model)
        scale2 = 1.0 / np.sqrt(d_ff)
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        h1 = np.dot(x, self.W1) + self.b1
        a1 = np.maximum(0, h1)  # ReLU
        out = np.dot(a1, self.W2) + self.b2
        cache = (x, h1, a1)
        return out, cache

    def backward(self, dout: np.ndarray, cache: Tuple) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        x, h1, a1 = cache
        dW2 = np.tensordot(a1, dout, axes=([0, 1], [0, 1]))
        db2 = np.sum(dout, axis=(0, 1))

        da1 = np.dot(dout, self.W2.T)
        dh1 = da1 * (h1 > 0)  # ReLU derivative

        dW1 = np.tensordot(x, dh1, axes=([0, 1], [0, 1]))
        db1 = np.sum(dh1, axis=(0, 1))

        dx = np.dot(dh1, self.W1.T)
        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return dx, grads


class TransformerBlock:
    """
    Single Transformer Decoder Block:
    x -> LayerNorm1 -> CausalSelfAttention -> Residual Add (x + Attn)
      -> LayerNorm2 -> FeedForward         -> Residual Add (x + FFN)
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        # 1. LN1 & Attention
        x_ln1, ln1_cache = self.ln1.forward(x)
        attn_out, attn_cache = self.attn.forward(x_ln1)
        x1 = x + attn_out  # Residual 1

        # 2. LN2 & FFN
        x_ln2, ln2_cache = self.ln2.forward(x1)
        ffn_out, ffn_cache = self.ffn.forward(x_ln2)
        x2 = x1 + ffn_out  # Residual 2

        cache = (x, x1, ln1_cache, attn_cache, ln2_cache, ffn_cache)
        return x2, cache

    def backward(self, dout: np.ndarray, cache: Tuple) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        x, x1, ln1_cache, attn_cache, ln2_cache, ffn_cache = cache

        # Backward through Residual 2 & FFN
        dx2 = dout
        dx_ffn = dx2.copy()
        dx1 = dx2.copy()

        dx_ln2, ffn_grads = self.ffn.backward(dx_ffn, ffn_cache)
        dx_from_ln2, dgamma2, dbeta2 = self.ln2.backward(dx_ln2, ln2_cache)
        dx1 += dx_from_ln2

        # Backward through Residual 1 & Attention
        dx_attn = dx1.copy()
        dx0 = dx1.copy()

        dx_ln1, attn_grads = self.attn.backward(dx_attn, attn_cache)
        dx_from_ln1, dgamma1, dbeta1 = self.ln1.backward(dx_ln1, ln1_cache)
        dx0 += dx_from_ln1

        grads = {
            "ln1_gamma": dgamma1, "ln1_beta": dbeta1,
            "ln2_gamma": dgamma2, "ln2_beta": dbeta2,
            **attn_grads,
            **ffn_grads
        }
        return dx0, grads


class TinyGPTLanguageModel:
    """
    Complete Tiny GPT-Style Language Model.
    
    Components:
    - Token Embedding: (vocab_size, d_model)
    - Positional Embedding: (context_length, d_model)
    - Transformer Block
    - Final Layer Normalization
    - Language Model Head: (d_model, vocab_size)
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int = 16,
        d_model: int = 32,
        n_heads: int = 2,
        d_ff: int = 64
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        np.random.seed(42)
        scale = 0.02
        self.W_te = np.random.randn(vocab_size, d_model) * scale
        self.W_pe = np.random.randn(context_length, d_model) * scale

        self.block = TransformerBlock(d_model, n_heads, d_ff)
        self.ln_f = LayerNorm(d_model)

        self.W_lm = np.random.randn(d_model, vocab_size) * scale
        self.b_lm = np.zeros(vocab_size)

    def forward(self, x_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple]:
        """
        Forward Pass:
        x_indices shape: (B, T)
        Returns: logits (B, T, V), probabilities (B, T, V), and cache
        """
        B, T = x_indices.shape
        assert T <= self.context_length, f"Sequence length {T} exceeds context_length {self.context_length}"

        # 1. Embeddings: Token + Positional
        tok_emb = self.W_te[x_indices]  # (B, T, D)
        pos_emb = self.W_pe[:T]         # (T, D) broadcast across B
        x0 = tok_emb + pos_emb          # (B, T, D)

        # 2. Transformer Block
        x_blk, blk_cache = self.block.forward(x0)

        # 3. Final LayerNorm
        x_final, ln_f_cache = self.ln_f.forward(x_blk)

        # 4. Output Logits (B, T, V)
        logits = np.dot(x_final, self.W_lm) + self.b_lm
        probs = stable_softmax(logits, axis=-1)

        cache = (x_indices, x0, blk_cache, ln_f_cache, x_final, probs)
        return logits, probs, cache

    def compute_loss(self, probs: np.ndarray, y_indices: np.ndarray, ignore_index: int = -100) -> float:
        """
        Calculates Cross-Entropy Loss over valid sequence positions:
            Loss = - (1 / N_valid) * sum(log(P(true_token)))
        Positions where y_indices == ignore_index (or < 0) are ignored (used in SFT to mask out prompts).
        """
        valid_mask = (y_indices != ignore_index) & (y_indices >= 0)
        num_valid = int(np.sum(valid_mask))
        if num_valid == 0:
            return 0.0

        B, T = y_indices.shape
        batch_idx = np.arange(B)[:, None]
        time_idx = np.arange(T)[None, :]
        safe_targets = np.where(valid_mask, y_indices, 0)
        correct_probs = probs[batch_idx, time_idx, safe_targets]
        loss = -np.sum(np.log(correct_probs + 1e-15) * valid_mask) / num_valid
        return float(loss)

    def get_token_log_probs(self, x_indices: np.ndarray, y_indices: np.ndarray, ignore_index: int = -100) -> np.ndarray:
        """
        Computes log probabilities of target tokens: log P(y_t | x_<=t).
        Positions with y_indices == ignore_index or < 0 are set to 0.0.
        Returns: (B, T) array of log probabilities.
        """
        _, probs, _ = self.forward(x_indices)
        B, T = y_indices.shape
        batch_idx = np.arange(B)[:, None]
        time_idx = np.arange(T)[None, :]
        valid_mask = (y_indices != ignore_index) & (y_indices >= 0)
        safe_targets = np.where(valid_mask, y_indices, 0)
        selected_probs = probs[batch_idx, time_idx, safe_targets]
        log_probs = np.log(selected_probs + 1e-15) * valid_mask
        return log_probs

    def backward_from_dlogits(self, dLogits: np.ndarray, cache: Tuple) -> Dict[str, np.ndarray]:
        """
        Backpropagation given explicit gradients on output Logits: dL/dz (B, T, V).
        Enables general objectives including Cross-Entropy, SFT, and Direct Preference Optimization (DPO).
        """
        x_indices, x0, blk_cache, ln_f_cache, x_final, probs = cache
        B, T, V = probs.shape

        # 1. LM Head backward
        dW_lm = np.tensordot(x_final, dLogits, axes=([0, 1], [0, 1]))
        db_lm = np.sum(dLogits, axis=(0, 1))
        dx_final = np.dot(dLogits, self.W_lm.T)

        # 2. Final LayerNorm backward
        dx_blk, dgamma_f, dbeta_f = self.ln_f.backward(dx_final, ln_f_cache)

        # 3. Transformer Block backward
        dx0, blk_grads = self.block.backward(dx_blk, blk_cache)

        # 4. Embeddings backward
        dW_te = np.zeros_like(self.W_te)
        np.add.at(dW_te, x_indices, dx0)
        dW_pe = np.zeros_like(self.W_pe)
        dW_pe[:T] = np.sum(dx0, axis=0)

        grads = {
            "W_te": dW_te,
            "W_pe": dW_pe,
            "W_lm": dW_lm,
            "b_lm": db_lm,
            "ln_f_gamma": dgamma_f,
            "ln_f_beta": dbeta_f,
            **blk_grads
        }
        return grads

    def backward(self, y_indices: np.ndarray, cache: Tuple, ignore_index: int = -100) -> Dict[str, np.ndarray]:
        """
        Full analytical backpropagation from Cross-Entropy Loss to all parameters.
        Correctly masks out positions where y_indices == ignore_index (or < 0).
        """
        x_indices, x0, blk_cache, ln_f_cache, x_final, probs = cache
        B, T, V = probs.shape

        valid_mask = (y_indices != ignore_index) & (y_indices >= 0)
        num_valid = max(int(np.sum(valid_mask)), 1)

        # Gradient on Logits: (P - 1_y) / num_valid
        dLogits = probs.copy()
        batch_idx = np.arange(B)[:, None]
        time_idx = np.arange(T)[None, :]
        safe_targets = np.where(valid_mask, y_indices, 0)
        dLogits[batch_idx, time_idx, safe_targets] -= 1.0
        dLogits *= valid_mask[:, :, None]
        dLogits /= num_valid

        return self.backward_from_dlogits(dLogits, cache)

    def get_params_and_grads(self, grads: Dict[str, np.ndarray]) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Pairs each parameter with its matching analytical gradient array.
        """
        param_map = [
            ("W_te", self.W_te, grads["W_te"]),
            ("W_pe", self.W_pe, grads["W_pe"]),
            ("ln1_gamma", self.block.ln1.gamma, grads["ln1_gamma"]),
            ("ln1_beta", self.block.ln1.beta, grads["ln1_beta"]),
            ("Wq", self.block.attn.Wq, grads["Wq"]),
            ("bq", self.block.attn.bq, grads["bq"]),
            ("Wk", self.block.attn.Wk, grads["Wk"]),
            ("bk", self.block.attn.bk, grads["bk"]),
            ("Wv", self.block.attn.Wv, grads["Wv"]),
            ("bv", self.block.attn.bv, grads["bv"]),
            ("Wo", self.block.attn.Wo, grads["Wo"]),
            ("bo", self.block.attn.bo, grads["bo"]),
            ("ln2_gamma", self.block.ln2.gamma, grads["ln2_gamma"]),
            ("ln2_beta", self.block.ln2.beta, grads["ln2_beta"]),
            ("W1", self.block.ffn.W1, grads["W1"]),
            ("b1", self.block.ffn.b1, grads["b1"]),
            ("W2", self.block.ffn.W2, grads["W2"]),
            ("b2", self.block.ffn.b2, grads["b2"]),
            ("ln_f_gamma", self.ln_f.gamma, grads["ln_f_gamma"]),
            ("ln_f_beta", self.ln_f.beta, grads["ln_f_beta"]),
            ("W_lm", self.W_lm, grads["W_lm"]),
            ("b_lm", self.b_lm, grads["b_lm"])
        ]
        return param_map

    def generate(
        self,
        tokenizer,
        prompt: str = "The ",
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        repetition_penalty: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        seed: Optional[int] = None
    ) -> str:
        """
        Autoregressive Generation:
        1. Encodes the prompt into tokens.
        2. Crops to max context_length.
        3. Predicts logits for the final position.
        4. Applies repetition penalty (if repetition_penalty > 1.0).
        5. Applies Temperature scaling and optional Top-K / Top-P (nucleus) filtering.
        6. Samples next token using stable probability distribution.
        7. Halts early if eos_token_id is sampled.
        8. Appends and repeats!
        """
        if seed is not None:
            np.random.seed(seed)

        if not prompt:
            prompt = "T"

        # Encode prompt
        encoded = tokenizer.encode(prompt, handle_unknown="replace")
        tokens = list(encoded)

        for _ in range(max_new_tokens):
            # Crop to context window if needed
            context_tokens = tokens[-self.context_length:]
            x_in = np.array([context_tokens], dtype=int)

            # Forward pass
            logits, probs, _ = self.forward(x_in)

            # Focus on the very last predicted token position (index -1)
            last_logits = logits[0, -1, :].copy()

            # Apply repetition penalty to recent context tokens
            if repetition_penalty > 1.0:
                recent = set(context_tokens[-8:])
                for t_id in recent:
                    if last_logits[t_id] > 0:
                        last_logits[t_id] /= repetition_penalty
                    else:
                        last_logits[t_id] *= repetition_penalty

            # Apply Top-K filtering if specified
            if top_k is not None and 0 < top_k < self.vocab_size:
                threshold = np.partition(last_logits, -top_k)[-top_k]
                last_logits[last_logits < threshold] = -1e9

            # Apply Temperature scaling
            scaled_logits = last_logits / max(temperature, 1e-5)

            # Apply Top-P (Nucleus) filtering if specified
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_indices = np.argsort(scaled_logits)[::-1]
                sorted_logits = scaled_logits[sorted_indices]
                sorted_probs = stable_softmax(sorted_logits, axis=-1)
                cumulative_probs = np.cumsum(sorted_probs)

                # Find tokens beyond the nucleus
                remove_mask = cumulative_probs > top_p
                remove_mask[1:] = remove_mask[:-1]
                remove_mask[0] = False

                indices_to_remove = sorted_indices[remove_mask]
                scaled_logits[indices_to_remove] = -1e9

            last_probs = stable_softmax(scaled_logits, axis=-1)

            # Sample next token
            next_token = int(np.random.choice(self.vocab_size, p=last_probs))
            tokens.append(next_token)

            # Stop if EOS token is encountered
            if eos_token_id is not None and next_token == eos_token_id:
                break

        return tokenizer.decode(tokens)

    def save_checkpoint(self, filepath: str):
        """
        Saves all weights and model hyperparameters to an .npz file (no pickle).
        """
        save_data = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "W_te": self.W_te,
            "W_pe": self.W_pe,
            "ln1_gamma": self.block.ln1.gamma,
            "ln1_beta": self.block.ln1.beta,
            "Wq": self.block.attn.Wq,
            "bq": self.block.attn.bq,
            "Wk": self.block.attn.Wk,
            "bk": self.block.attn.bk,
            "Wv": self.block.attn.Wv,
            "bv": self.block.attn.bv,
            "Wo": self.block.attn.Wo,
            "bo": self.block.attn.bo,
            "ln2_gamma": self.block.ln2.gamma,
            "ln2_beta": self.block.ln2.beta,
            "W1": self.block.ffn.W1,
            "b1": self.block.ffn.b1,
            "W2": self.block.ffn.W2,
            "b2": self.block.ffn.b2,
            "ln_f_gamma": self.ln_f.gamma,
            "ln_f_beta": self.ln_f.beta,
            "W_lm": self.W_lm,
            "b_lm": self.b_lm
        }
        np.savez(filepath, **save_data)
        print(f"Transformer checkpoint successfully saved to: {filepath}")

    def load_checkpoint(self, filepath: str, expected_vocab_size: Optional[int] = None):
        """
        Loads all weights and hyperparameters from an .npz file.
        Validates that the checkpoint's vocabulary size matches expected_vocab_size
        to prevent silent out-of-bounds errors or dimension mismatches.
        """
        data = np.load(filepath)
        ckpt_vocab = int(data["vocab_size"])

        if expected_vocab_size is not None and ckpt_vocab != expected_vocab_size:
            raise ValueError(
                f"Checkpoint vocabulary mismatch! Checkpoint '{filepath}' was trained with "
                f"vocab_size={ckpt_vocab}, but the active tokenizer has vocab_size={expected_vocab_size}. "
                f"Loading this checkpoint would cause token ID dimension errors. "
                f"Please use the matching checkpoint or retrain on the current vocabulary."
            )

        self.vocab_size = ckpt_vocab
        self.context_length = int(data["context_length"])
        self.d_model = int(data["d_model"])
        self.n_heads = int(data["n_heads"])
        self.d_ff = int(data["d_ff"])

        # Reconstruct internal blocks to match loaded architecture dimensions
        self.block = TransformerBlock(self.d_model, self.n_heads, self.d_ff)
        self.ln_f = LayerNorm(self.d_model)

        self.W_te = data["W_te"]
        self.W_pe = data["W_pe"]

        self.block.ln1.gamma = data["ln1_gamma"]
        self.block.ln1.beta = data["ln1_beta"]

        self.block.attn.Wq = data["Wq"]
        self.block.attn.bq = data["bq"]
        self.block.attn.Wk = data["Wk"]
        self.block.attn.bk = data["bk"]
        self.block.attn.Wv = data["Wv"]
        self.block.attn.bv = data["bv"]
        self.block.attn.Wo = data["Wo"]
        self.block.attn.bo = data["bo"]

        self.block.ln2.gamma = data["ln2_gamma"]
        self.block.ln2.beta = data["ln2_beta"]

        self.block.ffn.W1 = data["W1"]
        self.block.ffn.b1 = data["b1"]
        self.block.ffn.W2 = data["W2"]
        self.block.ffn.b2 = data["b2"]

        self.ln_f.gamma = data["ln_f_gamma"]
        self.ln_f.beta = data["ln_f_beta"]

        self.W_lm = data["W_lm"]
        self.b_lm = data["b_lm"]
        print(f"Transformer checkpoint successfully loaded from: {filepath} (vocab_size={self.vocab_size})")


class AdamOptimizer:
    """
    Adam / AdamW Optimizer (Adaptive Moment Estimation with Decoupled Weight Decay)
    from scratch in pure NumPy.
    
    Features:
    - Running first (mean) and second (variance) moment estimates.
    - Bias correction for early training steps.
    - Gradient clipping to prevent gradient explosions.
    - Decoupled L2 Weight Decay (AdamW): penalizes large weights on 2D weight matrices
      (embeddings, attention, FFN, LM head) while excluding 1D biases and LayerNorm parameters.
    """
    def __init__(
        self,
        learning_rate: float = 0.003,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        grad_clip: float = 1.0,
        weight_decay: float = 0.0
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.t = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def step(self, param_map_or_model, grads: Optional[Dict[str, np.ndarray]] = None):
        """
        Performs a single Adam/AdamW parameter update step.
        Accepts either:
        1. param_map: List[Tuple[name, param, grad]]
        2. model and grads: optimizer.step(model, grads)
        """
        if grads is not None:
            if hasattr(param_map_or_model, "get_params_and_grads"):
                param_map = param_map_or_model.get_params_and_grads(grads)
            elif hasattr(param_map_or_model, "get_param_map"):
                param_map = param_map_or_model.get_param_map(grads)
            else:
                raise AttributeError("Model must provide get_params_and_grads method.")
        else:
            param_map = param_map_or_model

        self.t += 1

        # 1. Compute global gradient norm for gradient clipping
        total_norm_sq = 0.0
        for name, param, grad in param_map:
            total_norm_sq += np.sum(grad ** 2)
        global_norm = np.sqrt(total_norm_sq)

        # Gradient clipping factor
        clip_scale = 1.0
        if self.grad_clip is not None and global_norm > self.grad_clip:
            clip_scale = self.grad_clip / (global_norm + 1e-12)

        # 2. Update each parameter
        for name, param, raw_grad in param_map:
            grad = raw_grad * clip_scale

            # Decoupled L2 Weight Decay (AdamW):
            # Only apply to 2D weight matrices (ndim >= 2: W_te, W_pe, Wq, Wk, Wv, Wo, W1, W2, W_lm).
            # 1D biases (b_*) and LayerNorm parameters (gamma, beta) are explicitly excluded!
            if self.weight_decay > 0.0 and param.ndim >= 2:
                param -= self.lr * self.weight_decay * param

            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)

            # Update biased first & second moment estimates
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first & second moment estimates
            m_hat = self.m[name] / (1.0 - (self.beta1 ** self.t))
            v_hat = self.v[name] / (1.0 - (self.beta2 ** self.t))

            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

