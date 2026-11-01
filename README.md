# Machine Learning & AI From Scratch (Pure Python & NumPy)

A progressive journey from a single linear neuron to a complete **GPT-Style Transformer Decoder with Byte-Pair Encoding (BPE), Supervised Fine-Tuning (SFT), and Direct Preference Optimization (DPO)**, built entirely from scratch with **pure Python & NumPy** (no PyTorch, no TensorFlow, no pretrained weights, no tokenizer libraries).

---

## 1. Project Structure

```text
ai model/
│
├── data/
│   ├── raw/                           # Clean English public-domain text files (11 files)
│   │   ├── aesop_fables_expanded.txt
│   │   ├── classic_prose_tales.txt
│   │   ├── daily_dialogues_and_adventures.txt
│   │   ├── fables_and_parables.txt
│   │   ├── historical_science_tales.txt
│   │   ├── literature_excerpts.txt
│   │   ├── nature_science_essays.txt
│   │   ├── people_and_society_tales.txt
│   │   ├── science_and_discovery.txt
│   │   ├── story_fables.txt
│   │   └── story_robot_adventures.txt
│   ├── clean_corpus.txt               # Combined & normalized corpus (29,370 chars, 5,165 words)
│   ├── instruction_data.jsonl         # [Stage 8] SFT instruction-response pairs
│   ├── preference_data.jsonl          # [Stage 8] DPO preference pairs (chosen vs rejected)
│   ├── safety_eval.jsonl              # [Stage 8] Safety & uncertainty evaluation probes
│   ├── tiny_story.txt                 # (Stage 3-5) Educational text corpus
│   ├── tiny_story_backup.txt          # Original 349-character text corpus backup
│   ├── student_scores.csv             # (Stage 1-2) 150 student records
│   └── generate_dataset.py            # Generates realistic student study habits and exam scores
│
├── src/
│   ├── __init__.py                    # Makes 'src' a Python package
│   ├── dataset.py                     # (Stage 1-2) CSV loader, train/test split, StandardScaler
│   ├── metrics.py                     # (Stage 1-2) MSE, RMSE, MAE, R² metrics
│   ├── model.py                       # (Stage 1) Linear Regression (single layer, analytical gradients)
│   ├── neural_network.py              # (Stage 2) Neural Network (Hidden Layer, ReLU, Backpropagation)
│   ├── tokenizer.py                   # [Stage 3, 7] CharTokenizer & BPETokenizer from scratch
│   ├── bigram_model.py                # (Stage 4) Bigram Language Model (Softmax, Cross-Entropy, Sampling)
│   └── transformer_model.py           # [Stage 5-8] Tiny GPT Transformer (Attention, FFN, LayerNorm, AdamW, Top-P)
│
├── prepare_dataset.py                 # (Stage 6, 8) Combines & audits .txt files from data/raw/
├── train.py                           # (Stage 1) Trains Linear Regression -> model_checkpoint.npz
├── train_nn.py                        # (Stage 2) Trains Neural Network   -> nn_checkpoint.npz
├── compare.py                         # (Stage 2) Compares Stage 1 vs Stage 2 side-by-side
├── predict.py                         # (Stage 1-2) Predictions for student score models
├── demo_tokenizer.py                  # (Stage 3) Demonstrates character-level tokenization
├── demo_bpe.py                        # (Stage 7) Demonstrates Byte-Pair Encoding subwords & merges
├── train_bigram.py                    # (Stage 4) Trains Bigram Model -> bigram_checkpoint.npz
├── generate.py                        # (Stage 4) Text generation with Bigram model
├── train_transformer.py               # [Stage 5-8] Trains Base Transformer (Zero-Leakage BPE, AdamW, Early Stopping)
├── train_sft.py                       # [Stage 8] Supervised Fine-Tuning (response-only loss masking)
├── train_dpo.py                       # [Stage 8] Direct Preference Optimization (analytical logit gradients)
├── evaluate_model.py                  # [Stage 8] Evaluates loss, perplexity, diversity, and safety probes
├── generate_transformer.py            # [Stage 5-8] Generates text (BPE/Char, Top-K, Top-P, Temperature)
├── test_transformer.py                # [Stage 5-8] Comprehensive 36-test suite & regression verification
│
├── model_checkpoint.npz               # Saved Stage 1 weights
├── nn_checkpoint.npz                  # Saved Stage 2 weights
├── bigram_checkpoint.npz              # Saved Stage 4 Bigram weights
├── transformer_checkpoint.npz         # Saved Stage 6 character model weights (60-char vocab)
├── transformer_bpe_checkpoint.npz     # [Stage 8] Saved Base BPE model weights (141-token vocab)
├── transformer_sft_checkpoint.npz     # [Stage 8] Saved SFT Instruction-Tuned weights
├── transformer_dpo_checkpoint.npz     # [Stage 8] Saved DPO Preference-Aligned weights
├── bpe_vocab.json                     # [Stage 8] BPE vocabulary (141 tokens) and 80 merge rules
├── transformer_vocab.json             # Stage 6 vocabulary mapping (60 tokens)
├── tokenizer_vocab.json               # Vocabulary mapping for Stage 5 / Stage 3
├── bigram_vocab.json                  # Preserved 35-token vocabulary for Stage 4 Bigram model
├── eval_report.json                   # [Stage 8] Comprehensive evaluation metrics and sample outputs
└── README.md                          # Complete multi-stage documentation
```

---

## 2. Complete Evolution Roadmap

| Stage | Model Type | Core Concept Learned | Run Command |
| :---: | :--- | :--- | :--- |
| **Stage 1** | **Linear Regression** | Weights, Bias, Mean Squared Error, Analytical Gradient Descent | `python train.py` |
| **Stage 2** | **Neural Network (MLP)** | Hidden Layers, Non-linear ReLU, Analytical Backpropagation Chain Rule | `python train_nn.py` |
| **Stage 3** | **Character Tokenizer** | Character Vocabulary, Integer Token IDs, Round-Trip Fidelity | `python demo_tokenizer.py` |
| **Stage 4** | **Bigram Language Model** | Transition Matrix, Softmax, Cross-Entropy Loss, Temperature Sampling | `python train_bigram.py` |
| **Stage 5** | **Tiny GPT Transformer** | Causal Multi-Head Self-Attention, LayerNorm, Residuals, Backpropagation | `python train_transformer.py` |
| **Stage 6** | **Dataset & Regularization** | Clean Text Corpus Pipeline, AdamW Weight Decay, Early Stopping, Top-K | `python prepare_dataset.py` |
| **Stage 7** | **Byte-Pair Encoding (BPE)** | Subword Tokenization, Statistical Merges, Sequence Compression, BPE GPT | `python demo_bpe.py` |
| **Stage 8** | **Reliability, SFT & DPO** | Zero-Leakage Pipeline, Top-P, Response Masking SFT, Direct Preference Optimization | `python evaluate_model.py` |

---

## 3. Stage 8: Reliability, SFT & Preference Optimization

### A. 10-Point Engineering Correctness Audit
1. **Causal Attention Masking**: Attention scores have an upper-triangular mask of `-1e9` ensuring position $i$ attends only to positions $j \le i$. Test 3 & 4 verify strictly 0.0 attention to future tokens.
2. **Positional & Token Embeddings**: Separate token embedding $W_{\text{te}}$ and learned positional embedding $W_{\text{pe}}$ tables. Test 7 verifies analytical gradients match numerical gradients with error $< 3.5 \times 10^{-9}$.
3. **Numerically Stable Softmax**: Subtracts $\max(x)$ along the target axis before exponentiation to eliminate positive overflow ($e^{1000} \rightarrow \text{inf}$).
4. **Cross-Entropy Loss with Target Masking**: Supports `ignore_index = -100` to mask out instruction prompt tokens during SFT so the model only learns response generation.
5. **Analytical Backpropagation**: Exact closed-form gradients derived through Self-Attention, LayerNorm, Residual Additions, ReLU FFN, and Output Projections.
6. **Gradient Clipping**: Computes global $\ell_2$ parameter gradient norm and rescales updates when norm exceeds threshold (`1.0`).
7. **Weight Initialization**: Gaussian initialization with $\sigma = 0.02$ for embeddings/LM head (matching OpenAI GPT-2) and Xavier $1/\sqrt{d_{\text{in}}}$ for linear layers.
8. **Decoupled L2 Weight Decay (AdamW)**: Regularizes 2D projection matrices ($W_q, W_k, W_v, W_o, W_1, W_2, W_{\text{te}}, W_{\text{lm}}$) while strictly preserving 1D biases and LayerNorm parameters.
9. **Zero-Leakage Data Pipeline**: The raw corpus is split into 85% train and 15% validation **before** fitting the tokenizer. Vocabulary and merge rules are learned exclusively from the training split.
10. **Vocabulary Compatibility Guard**: `load_checkpoint` enforces strict `expected_vocab_size` checks to prevent silent matrix shape mismatches.

### B. Investigation: Why Was `"people"` Split into Individual Characters?
In Stage 7, the word `"people"` was tokenized as 6 separate characters `['p', 'e', 'o', 'p', 'l', 'e']`.
- **Reason**: In the 22k-character corpus, `"people"` only appeared **2 times**. With a merge budget of 60 merges, high-frequency pairs (`'e '`, `' t'`, `'he '`) consumed all slots.
- **Solution**: In Stage 8, we expanded the corpus to 29,370 characters, increasing occurrences of `"people"` to 14, and expanded the merge budget to 80 merges. The tokenizer learned subwords `'op'` and `'le'`, reducing `"people"` to `['p', 'e', 'op', 'le']`.

### C. Top-P (Nucleus) Sampling
Unlike Top-K (which retains a fixed number of candidates), Top-P dynamically chooses the smallest set of tokens whose cumulative probability exceeds threshold $p$ (e.g. 0.9):
1. Sort logits in descending order.
2. Compute cumulative softmax probabilities.
3. Zero out logits beyond cumulative probability threshold $p$.
4. Renormalize and sample.
This removes the low-probability tail while adapting to both sharp and flat distributions.

### D. Supervised Fine-Tuning (SFT / Instruction Tuning)
- **Dataset**: `data/instruction_data.jsonl` contains 15 curated English instruction-response pairs.
- **Response-Only Loss Masking**: Prompts are formatted as `Instruction:\n{instruction}\n\nResponse:\n`. Target tokens for prompt positions are set to `-100`, so the loss is computed **strictly on response tokens**.
- **Results**: Held-out validation instruction loss dropped from `3.3009` (Base Model) to `2.9569` (SFT Model), learning to output response sentences rather than raw narrative prose.

### E. Direct Preference Optimization (DPO)
DPO (Rafailov et al., 2023) directly optimizes preference pairs $(x, y_w, y_l)$ using the closed-form objective:
$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = \log\left(1 + \exp\left(-\beta \left[ \left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}\right) - \left(\log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right) \right]\right)\right)$$
- **Analytical Logit Gradients**:
  $$dZ_w = +\beta \sigma(-M) \cdot (p_w - \mathbf{1}_{y_w}) / N_w$$
  $$dZ_l = -\beta \sigma(-M) \cdot (p_l - \mathbf{1}_{y_l}) / N_l$$
- **Results**: DPO loss dropped from $0.6937 \approx \ln(2)$ to $0.4323$, the preference margin increased from $-0.0010$ to $+0.6170$, and preference accuracy reached 100%.

---

## 4. Stage 8 Benchmark Results Across Checkpoints

| Metric | Base Model (`transformer_bpe`) | SFT Model (`transformer_sft`) | DPO Model (`transformer_dpo`) |
| :--- | :---: | :---: | :---: |
| **Vocabulary Size** | 141 subwords | 141 subwords | 141 subwords |
| **Training Loss** | 2.8942 | 3.1664 | 4.6680 |
| **Validation Loss** | **3.3580** | 3.5281 | 5.0196 |
| **Validation Perplexity** | **28.73** | 34.06 | 151.35 |
| **Distinct-1 (Unigram Diversity)** | 0.675 | 0.619 | 0.655 |
| **Distinct-2 (Bigram Diversity)** | 0.842 | 0.843 | **0.879** |
| **Format Adaptation** | Raw Prose | Answers Prompts (`"It stare..."`) | Aligned (`"Dsteady..."`) |
| **Test Suite** | **36 passing tests** | **36 passing tests** | **36 passing tests** |

> [!IMPORTANT]
> **Understanding Small Model Limitations & Safety**:
> Commercial models (GPT-4, Claude) possess billions of parameters and train on trillions of tokens. Our model is built from scratch with ~25,000 parameters and trained on 29,370 characters (~5,000 words).
> - It learns real English subwords, basic syllable transitions, and format alignment.
> - However, it **does not possess factual world knowledge, reasoning, or human-level safety understanding**.
> - In the safety audit on sensitive probes, it generates pseudo-English prose from its learned distribution. Safety in language models requires immense scale, external verifiers, and extensive alignment—it cannot be faked with small prompts.

---

## 5. Commands to Run Stage 8

### 1. Compile and Audit Expanded Corpus
```powershell
python prepare_dataset.py
```

### 2. Run the Full 36-Test Suite
```powershell
python test_transformer.py
```

### 3. Train the Zero-Leakage Base Transformer
```powershell
python train_transformer.py --tokenizer bpe --num-merges 80 --epochs 35 --patience 6
```

### 4. Run Supervised Fine-Tuning (SFT)
```powershell
python train_sft.py --epochs 20 --lr 0.001
```

### 5. Run Direct Preference Optimization (DPO)
```powershell
python train_dpo.py --epochs 15 --beta 0.1 --lr 0.0005
```

### 6. Run Comprehensive Reliability Evaluation
```powershell
python evaluate_model.py
```

### 7. Generate Text with Top-P Nucleus Sampling
```powershell
# Generate from Base Model with Top-P = 0.9
python generate_transformer.py --prompt "Once upon a time " --temp 0.6 --top-p 0.9 --length 40

# Generate from SFT Model
python generate_transformer.py --checkpoint transformer_sft_checkpoint.npz --prompt "Instruction:\nSummarize the lesson.\n\nResponse:\n" --temp 0.6 --top-p 0.9
```

### 8. Regression Verification (Stages 1–7)
```powershell
python predict.py --hours 8.0 --questions 50
python generate.py --prompt "The " --length 30
python demo_bpe.py
```
