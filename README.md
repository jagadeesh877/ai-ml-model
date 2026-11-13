# Machine Learning & AI From Scratch (Pure Python & NumPy)
# 🚀 Pure Python & NumPy GPT Transformer: From Scratch

A progressive journey from a single linear neuron to a complete **GPT-Style Transformer Decoder with Byte-Pair Encoding (BPE), Supervised Fine-Tuning (SFT), and Direct Preference Optimization (DPO)**, built entirely from scratch with **pure Python & NumPy** (no PyTorch, no TensorFlow, no pretrained weights, no tokenizer libraries).
[![Python 3.14+](https://img.shields.io/badge/Python-3.14%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Engine: Pure NumPy](https://img.shields.io/badge/Engine-Pure%20NumPy%20(Zero%20PyTorch)-green.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Tests Passing](https://img.shields.io/badge/Test%20Suite-39%20Passing%20(100%25)-success.svg)](file:///test_transformer.py)
[![Parameters](https://img.shields.io/badge/Parameters-66%2C737%20Weights-purple.svg)](file:///src/transformer_model.py)
[![BPE Vocab](https://img.shields.io/badge/BPE%20Vocab-241%20Subwords-orange.svg)](file:///bpe_vocab.json)
[![Compression](https://img.shields.io/badge/Compression-2.01x%20(50.4%25)-cyan.svg)](file:///src/tokenizer.py)
[![Best Val Loss](https://img.shields.io/badge/Best%20Val%20Loss-3.4557%20(PPL%2031.68)-emerald.svg)](file:///checkpoints_stride_4/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An educational and production-grade implementation of a **GPT-Style Causal Transformer Decoder with Byte-Pair Encoding (BPE), Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Live Visual Analytics**, written entirely from scratch using only **Python standard libraries and NumPy** (no PyTorch, no TensorFlow, no HuggingFace, no external tokenizers).

---

## 1. Project Structure
## 📊 1. Live Visual Analytics & Telemetry Dashboard

This repository features real-time telemetry, automated checkpoint auditing, and visual reporting.

> [!TIP]
> **Launch Live Analytics Dashboard anytime**:
> ```powershell
> # View colorful ASCII terminal dashboard & compile analytics.html:
> python show_analytics.py
>
> # Automatically open the interactive web dashboard in your browser:
> python show_analytics.py --open
> ```

```text
==============================================================================
      TINY GPT TRANSFORMER: LIVE ANALYTICS & PERFORMANCE DASHBOARD      
      Pure Python & NumPy | Zero External Dependencies | From Scratch   
==============================================================================
  [1] MODEL SCALE & RECEPTIVE FIELD:
      - Total Parameters:   66,737 trainable weights
      - Context Window (T): 32 tokens (~64 characters)
      - Attention Heads:    4 heads (head_dim = 16)
      - Hidden Dim (d_model): 64 | Feed-Forward Dim (d_ff): 128

  [2] TOKENIZER & GENERALIZATION TELEMETRY:
      - BPE Vocabulary:     241 subwords (180 merges, 61 base chars)
      - Compression Ratio:  2.01x (50.3% sequence length reduction)
      - Whole Words Mapped: 49 words with leading whitespace
      - Clean Corpus Size:  71,603 chars (12,657 words)
      - Best Validation:    Loss 3.4557 | Perplexity 31.68
      - Unit & Regr Tests:  39 / 39 PASSING (100% test pass rate)
------------------------------------------------------------------------------
  [3] LOSS TRAJECTORY: STRIDE 4 (ENHANCED STORIES) vs STRIDE 6 vs STRIDE 2
------------------------------------------------------------------------------
   Epoch  | Stride 4 Train | Stride 4 Val  | Stride 6 Val  | Stride 2 Val 
------------------------------------------------------------------------------
     1    |     4.2341     |    3.7209     |    4.3165     |    3.9172    
     2    |     3.2874     |    3.4982     |    3.9419     |    4.0690    
     3    |     2.8684     |   3.4557 *    |    3.9307     |    4.3906    
     4    |     2.5899     |    3.4868     |    3.9092     |    4.6572    
     5    |     2.3883     |    3.5064     |    3.9075     |    4.7778    
     6    |     2.2858     |    3.5220     |       -       |       -      
------------------------------------------------------------------------------
  Stride 4 Trend: Train Loss: ..:-=@ (4.2341 -> 2.2858)
                  Val Loss:   ::..:@ (3.7209 -> 3.4557)
==============================================================================
```

---

## 📈 2. Training Curves & Convergence Dynamics

### Loss Trajectory Chart (Mermaid Interactive)

```mermaid
xychart-beta
    title "Training vs Validation Loss (Stride 6 Balanced Run)"
    x-axis ["Epoch 1", "Epoch 2", "Epoch 3", "Epoch 4", "Epoch 5"]
    y-axis "Cross-Entropy Loss" 2.5 --> 4.8
    line "Stride 6 Train Loss" [4.7436, 3.7819, 3.2904, 2.9743, 2.8138]
    line "Stride 6 Val Loss (Best)" [4.3165, 3.9419, 3.9307, 3.9092, 3.9075]
```

### Comprehensive Checkpoint Evaluation Matrix

All checkpoints were evaluated with fixed random seed (`42`), `temperature = 0.6`, `top_p = 0.9`, and `repetition_penalty = 1.1`:

| Experiment Run | Epoch | Train Loss | Train PPL | Val Loss | Val PPL | Real Words % | Distinct-1 Diversity | Generalization Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stride 6 (Balanced)** | **Epoch 1** | 4.1708 | 64.77 | 4.3165 | 74.93 | 43.5% | 0.667 | Initial alignment |
| **Stride 6 (Balanced)** | **Epoch 2** | 3.4204 | 30.58 | 3.9419 | 51.52 | 57.8% | 0.688 | Syntax structure forms |
| **Stride 6 (Balanced)** | **Epoch 3** | 3.0709 | 21.56 | 3.9307 | 50.94 | 44.4% | 0.778 | Clause boundaries stabilize |
| **Stride 6 (Balanced)** | **Epoch 4** | 2.8430 | 17.17 | 3.9092 | 49.86 | 57.8% | 0.812 | Vocabulary accuracy rises |
| **Stride 6 (Balanced)** | **Epoch 5** | **2.7575** | **15.76** | **3.9075** | **49.77** | **60.3%** | **0.778** | **🏆 BEST CHECKPOINT OVERALL** |
| | | | | | | | | |
| **Stride 2 (High Overlap)** | **Epoch 1** | 3.2736 | 26.41 | 3.9172 | 50.26 | 48.3% | 0.767 | Rapid initial drop |
| **Stride 2 (High Overlap)** | **Epoch 2** | 2.5849 | 13.26 | 4.0690 | 58.50 | 64.6% | 0.677 | Highest raw word rate |
| **Stride 2 (High Overlap)** | **Epoch 3** | 2.1390 | 8.49 | 4.3906 | 80.69 | 68.3% | 0.778 | Overfitting begins |
| **Stride 2 (High Overlap)** | **Epoch 4** | 1.8184 | 6.16 | 4.6572 | 105.34 | 71.9% | 0.766 | Divergence on unseen text |
| **Stride 2 (High Overlap)** | **Epoch 5** | 1.6840 | 5.39 | 4.7778 | 118.84 | 78.3% | 0.754 | Severe cross-entropy penalty |

> [!IMPORTANT]
> **Why Stride 6 Outperformed Stride 2**:
> In Stride 2, consecutive 32-token sequences overlapped by **30 tokens (93.75% overlap)**, causing the model to see every training phrase 16 times per epoch. The model memorized the training split and assigned extreme confidence ($P \to 1.0$), which severely penalized unseen vocabulary in the held-out validation stories ($P \approx 10^{-4} \implies -\log(P) \approx 9.2$). Stride 6 eliminated this redundancy, enabling smooth, steady validation improvement down to **3.9075** (Perplexity **49.77**).

---

## 🏛️ 3. Transformer Model Architecture & Dataflow

The model implements a standard GPT-2 style decoder-only transformer with causal self-attention, layer normalization, residual skip-connections, and learned positional embeddings.

```mermaid
flowchart TD
    Input["Input Prompt String (e.g. 'once upon time')"] --> Pretok["Leading-Space Regex Pre-Tokenization (' ?\w+')"]
    Pretok --> BPE["Byte-Pair Encoding (241 Vocabulary IDs)"]
    BPE --> Crop["Context Window Crop (T ≤ 32 Tokens)"]
    
    subgraph TransformerCore ["Tiny GPT Decoder Block (Pure NumPy)"]
        Crop --> TokEmb["Token Embedding W_te (241 × 64)"]
        Crop --> PosEmb["Positional Embedding W_pe (32 × 64)"]
        TokEmb --> AddEmb["Sum Embeddings: x0 = W_te[x] + W_pe[:T]"]
        PosEmb --> AddEmb
        
        AddEmb --> LN1["Layer Normalization 1 (d=64)"]
        LN1 --> MHA["Causal Multi-Head Attention (4 Heads, head_dim=16)"]
        MHA --> Res1["Residual Connection 1: x1 = x0 + Attn(LN1(x0))"]
        AddEmb -. Skip 1 .-> Res1
        
        Res1 --> LN2["Layer Normalization 2 (d=64)"]
        LN2 --> FFN["Feed-Forward Network: ReLU(x @ W1 + b1) @ W2 + b2 (d_ff=128)"]
        FFN --> Res2["Residual Connection 2: x2 = x1 + FFN(LN2(x1))"]
        Res1 -. Skip 2 .-> Res2
        
        Res2 --> LNF["Final Layer Normalization (d=64)"]
    end
    
    LNF --> Head["LM Projection Head W_lm (64 × 241) + b_lm"]
    Head --> Logits["Vocabulary Logits (B, T, 241)"]
    Logits --> RepPen["Repetition Penalty (scale=1.1)"]
    RepPen --> Temp["Temperature Scaling (temp=0.6)"]
    Temp --> TopP["Top-P Nucleus Sampling (p=0.90)"]
    TopP --> Sample["Next-Token Probabilistic Sample"]
    Sample --> Output["Decoded Human Text Stream"]
```

---

## 🔬 4. Exact Mathematical Parameter Breakdown

The model contains exactly **66,737 trainable floating-point weights**:

| Layer / Parameter Matrix | Exact Array Dimensions | Parameter Count | % of Total Model | Function in Architecture |
| :--- | :---: | :---: | :---: | :--- |
| **Token Embeddings ($W_{\text{te}}$)** | `(241, 64)` | 15,424 | 23.1% | Maps subword integer IDs to continuous 64-dim latent vectors |
| **Positional Embeddings ($W_{\text{pe}}$)** | `(32, 64)` | 2,048 | 3.1% | Encodes sequence order across the 32-token receptive field |
| **Attention LayerNorm ($\gamma_1, \beta_1$)** | `(64,) + (64,)` | 128 | 0.2% | Normalizes activations across features before self-attention |
| **Attention Queries ($W_q, b_q$)** | `(64, 64) + (64,)` | 4,160 | 6.2% | Projects token features into query space for 4 attention heads |
| **Attention Keys ($W_k, b_k$)** | `(64, 64) + (64,)` | 4,160 | 6.2% | Projects token features into key space for scaled dot-product |
| **Attention Values ($W_v, b_v$)** | `(64, 64) + (64,)` | 4,160 | 6.2% | Encodes contextual information aggregated by attention |
| **Attention Output ($W_o, b_o$)** | `(64, 64) + (64,)` | 4,160 | 6.2% | Recombines 4 attention heads back into 64-dim model space |
| **FFN LayerNorm ($\gamma_2, \beta_2$)** | `(64,) + (64,)` | 128 | 0.2% | Normalizes residual features before non-linear feed-forward |
| **FFN Expansion ($W_1, b_1$)** | `(64, 128) + (128,)` | 8,320 | 12.5% | Expands representation to 128 dimensions with ReLU non-linearity |
| **FFN Contraction ($W_2, b_2$)** | `(128, 64) + (64,)` | 8,256 | 12.4% | Projects non-linear activations back to 64-dim model space |
| **Final LayerNorm ($\gamma_f, \beta_f$)** | `(64,) + (64,)` | 128 | 0.2% | Stabilizes hidden states before vocabulary projection |
| **LM Output Head ($W_{\text{lm}}, b_{\text{lm}}$)** | `(64, 241) + (241,)` | 15,665 | 23.5% | Computes final logits across all 241 vocabulary subwords |
| **Total Model Weights** | **Pure NumPy** | **66,737** | **100.0%** | **Complete GPT-Style Transformer Decoder** |

---

## 🗺️ 5. Complete Evolution Roadmap (Stages 1 through 8+)

```mermaid
flowchart LR
    S1["Stage 1: Linear Regression"] --> S2["Stage 2: Neural Net (MLP)"]
    S2 --> S3["Stage 3: Char Tokenizer"]
    S3 --> S4["Stage 4: Bigram Model"]
    S4 --> S5["Stage 5: Tiny GPT"]
    S5 --> S6["Stage 6: AdamW & Data"]
    S6 --> S7["Stage 7: Subword BPE"]
    S7 --> S8["Stage 8: SFT & DPO"]
    S8 --> S8P["Stage 8+: Scaled 66.7k Model"]
```

| Stage | Model Type | Core Concept Learned | Best Loss / Metric | Test Suite | Run Command |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **Stage 1** | **Linear Regression** | Weights, Biases, MSE Loss, Closed-Form Normal Equations | MSE 32.14 | Test 13 | `python train.py` |
| **Stage 2** | **Neural Network (MLP)** | Hidden Layers, Non-linear ReLU, Analytical Backprop Chain Rule | MSE 11.28 | Test 13 | `python train_nn.py` |
| **Stage 3** | **Character Tokenizer** | Vocab Dictionaries, Integer Token IDs, 100% Round-Trip Fidelity | Lossless | Test 14 | `python demo_tokenizer.py` |
| **Stage 4** | **Bigram Language Model** | Transition Probability Matrix, Softmax, Cross-Entropy, Sampling | CE 2.68 | Test 15 | `python train_bigram.py` |
| **Stage 5** | **Tiny GPT Transformer** | Causal Self-Attention, Positional Embeddings, LayerNorm, Residuals | CE 2.36 | Tests 1-12 | `python train_transformer.py` |
| **Stage 6** | **Dataset & Regularization** | Clean Text Corpus Pipeline, Decoupled AdamW Weight Decay, Top-K | CE 2.02 | Tests 16-25 | `python prepare_dataset.py` |
| **Stage 7** | **Byte-Pair Encoding (BPE)** | Subword Tokenization, Statistical Pair Merging, Sequence Compression | CE 3.28 | Tests 26-30 | `python demo_bpe.py` |
| **Stage 8** | **Reliability, SFT & DPO** | Zero-Leakage Split, Masked Cross-Entropy, Direct Preference Opt | Margin +0.61 | Tests 31-36 | `python train_dpo.py` |
| **Stage 8+** | **Scaled Model & Analytics** | 66.7k Params, Leading-Space Pre-Tokenization, Stride Diagnostics | **Val 3.9075** | **Tests 37-39** | `python show_analytics.py` |

---

## 💬 6. Text Generation Benchmark (Before vs After)

Sampling settings: `temperature = 0.6`, `top_p = 0.9`, `repetition_penalty = 1.1`, `max_new_tokens = 40`, `seed = 42`.

### Prompt 1: `"once upon time"`
* **Stage 7 / Early Stage 8 (Character/Naive BPE)**:
  ```text
  once upon timether.
  Thentere an of live besteady comped the toion every win of the
  ```
  *(Defect: Word-boundary fusion error caused by trailing spaces in tokens: "time" + "there" → "timether")*
* **Stage 8+ Best Checkpoint (Leading-Space BPE, 66.7k Model)**:
  ```text
  once upon time, "Whying a pat the quiet wheres water, and construs, wheer the gooden a fin
  ```
  *(Improvement: Clean word boundaries, proper punctuation, dialogue quotation marks, zero spacing fusion)*

### Prompt 2: `"The brave knight "`
* **Stage 8 Baseline (Mode-Collapsed)**:
  ```text
  The brave knight play whoke its bortoils. Thentring the where astreed to himby f
  ```
* **Stage 8+ Best Checkpoint**:
  ```text
  The brave knight Tornown, time do nothing ary mathine. The found meady turning across
  ```

### Prompt 3: `"In the middle of the garden "`
* **Stage 8+ Best Checkpoint**:
  ```text
  In the middle of the garden Riding As Where quiet acre agains, the wooden will reace. The lived boy c
  ```

### Prompt 4: `"A clever fox noticed "`
* **Stage 8+ Best Checkpoint**:
  ```text
  A clever fox noticed raveler, the impered who flows. He wonder the rave, byward hearms of the
  ```

### Prompt 5: `"Long ago in a distant kingdom "`
* **Stage 8+ Best Checkpoint**:
  ```text
  Long ago in a distant kingdom race of sclient and lack and scient and hughed water disphee them. The Grass
  ```

---

## 📁 7. Project File Structure

```text
ai model/
│
├── data/
│   ├── raw/                           # Clean English public-domain text files (11 files)
│   ├── raw/                           # 17 Curated English public-domain text files (45,727 chars)
│   │   ├── aesop_fables_expanded.txt
│   │   ├── andersen_and_classic_tales.txt
│   │   ├── classic_fairytales_and_fables.txt
│   │   ├── classic_prose_tales.txt
│   │   ├── daily_dialogues_and_adventures.txt
│   │   ├── fables_and_parables.txt
│   │   ├── historical_science_tales.txt
│   │   ├── inventors_and_craftsmen.txt
│   │   ├── literature_excerpts.txt
│   │   ├── nature_and_wildlife_chronicles.txt
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
│   │   ├── story_robot_adventures.txt
│   │   ├── tales_of_adventure_and_mystery.txt
│   │   └── world_explorers_and_tales.txt
│   ├── clean_corpus.txt               # Unified & normalized training corpus (45,727 chars, 8,055 words)
│   ├── instruction_data.jsonl         # [Stage 8] Curated instruction-response pairs for SFT
│   ├── preference_data.jsonl          # [Stage 8] Preference pairs for Direct Preference Optimization
│   ├── safety_eval.jsonl              # [Stage 8] Probes for safety and uncertainty auditing
│   ├── tiny_story.txt                 # (Stage 3-5) Initial educational text dataset
│   └── student_scores.csv             # (Stage 1-2) 150 student exam records
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
│   ├── __init__.py                    # Python package initializer
│   ├── dataset.py                     # CSV dataset loader, train/test split, StandardScaler
│   ├── metrics.py                     # MSE, RMSE, MAE, R² metrics
│   ├── model.py                       # (Stage 1) Linear Regression (analytical normal equations)
│   ├── neural_network.py              # (Stage 2) Multilayer Perceptron (ReLU, backprop chain rule)
│   ├── tokenizer.py                   # (Stage 3, 7, 8+) CharTokenizer & Leading-Space BPETokenizer
│   ├── bigram_model.py                # (Stage 4) Bigram Language Model
│   └── transformer_model.py           # (Stage 5-8+) Tiny GPT Transformer (Causal Attn, LayerNorm, AdamW)
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
├── train_transformer.py               # [Stage 5-8+] Trains Transformer (Zero-Leakage BPE, AdamW, Cosine LR)
├── generate_transformer.py            # [Stage 5-8+] Text generator with Top-P, Top-K, Sentence Completion
├── evaluate_model.py                  # Evaluates validation loss, perplexity, diversity, and safety probes
├── evaluate_all_checkpoints.py        # Compares multiple checkpoint runs across epochs
├── show_analytics.py                  # 📊 Live terminal analytics & HTML dashboard compiler
├── analytics.html                     # 🌐 Standalone interactive dark-mode web dashboard
├── test_transformer.py                # 🧪 Comprehensive 39-test regression and unit test suite
├── train_sft.py                       # [Stage 8] Supervised Fine-Tuning (response-only loss masking)
├── train_dpo.py                       # [Stage 8] Direct Preference Optimization (analytical logit gradients)
│
├── checkpoints_stride_4/              # 🏆 Latest Best Run (Stride 4, 71k Story Corpus, Val Loss: 3.4557)
│   ├── checkpoint_epoch_1.npz
│   ├── checkpoint_epoch_2.npz
│   ├── checkpoint_epoch_3.npz         # Best validation checkpoint (Val Loss: 3.4557, PPL: 31.68)
│   ├── checkpoint_epoch_4.npz
│   ├── checkpoint_epoch_5.npz
│   ├── checkpoint_epoch_6.npz         # Deepest training checkpoint (Train Loss: 2.2858)
│   ├── checkpoint_best_val.npz
│   └── training_history.json
│
├── checkpoints_stride_6/              # Previous baseline checkpoints (Stride 6, Val Loss: 3.9075)
├── checkpoints_stride_2/              # High-overlap comparison checkpoints (Stride 2)
│
├── model_checkpoint.npz               # Saved Stage 1 Linear Regression weights
├── nn_checkpoint.npz                  # Saved Stage 2 Neural Network weights
├── bigram_checkpoint.npz              # Saved Stage 4 Bigram weights
├── transformer_checkpoint.npz         # Saved Stage 6 character model weights
├── transformer_bpe_checkpoint.npz     # 🚀 Active Production Checkpoint (241 subwords, 66,737 parameters)
├── transformer_sft_checkpoint.npz     # Saved SFT Instruction-Tuned weights
├── transformer_dpo_checkpoint.npz     # Saved DPO Preference-Aligned weights
├── bpe_vocab.json                     # Active BPE vocabulary (241 tokens, 180 merge rules)
├── transformer_vocab.json             # Stage 6 vocabulary mapping (60 tokens)
├── tokenizer_vocab.json               # Vocabulary mapping for Stage 5 / Stage 3
├── bigram_vocab.json                  # Preserved vocabulary for Stage 4 Bigram model
├── checkpoint_evaluation_results.json # Checkpoint telemetry and diagnostic results
├── eval_report.json                   # Comprehensive evaluation metrics and sample outputs
└── README.md                          # Complete project documentation and live analytics
```

---

## 2. Complete Evolution Roadmap
## 🛠️ 8. Command Reference

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
### Live Analytics & Dashboard
```powershell
# Show terminal telemetry dashboard and export analytics.html
python show_analytics.py

---
# Generate and launch the interactive web dashboard in your browser
python show_analytics.py --open

## 3. Stage 8: Reliability, SFT & Preference Optimization
# Output raw metrics in structured JSON format
python show_analytics.py --json
```

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
### Text Generation with Top-P Nucleus Sampling
```powershell
# Generate text from active checkpoint (Stride 6 Epoch 5)
python generate_transformer.py --prompt "once upon time" --temp 0.6 --top-p 0.9 --length 40

### B. Investigation: Why Was `"people"` Split into Individual Characters?
In Stage 7, the word `"people"` was tokenized as 6 separate characters `['p', 'e', 'o', 'p', 'l', 'e']`.
- **Reason**: In the 22k-character corpus, `"people"` only appeared **2 times**. With a merge budget of 60 merges, high-frequency pairs (`'e '`, `' t'`, `'he '`) consumed all slots.
- **Solution**: In Stage 8, we expanded the corpus to 29,370 characters, increasing occurrences of `"people"` to 14, and expanded the merge budget to 80 merges. The tokenizer learned subwords `'op'` and `'le'`, reducing `"people"` to `['p', 'e', 'op', 'le']`.
# Run a temperature sweep across focused, balanced, and creative scales
python generate_transformer.py --prompt "The brave knight " --temp-sweep --length 40

### C. Top-P (Nucleus) Sampling
Unlike Top-K (which retains a fixed number of candidates), Top-P dynamically chooses the smallest set of tokens whose cumulative probability exceeds threshold $p$ (e.g. 0.9):
1. Sort logits in descending order.
2. Compute cumulative softmax probabilities.
3. Zero out logits beyond cumulative probability threshold $p$.
4. Renormalize and sample.
This removes the low-probability tail while adapting to both sharp and flat distributions.
# Interactive CLI chat session
python generate_transformer.py
```

### D. Supervised Fine-Tuning (SFT / Instruction Tuning)
- **Dataset**: `data/instruction_data.jsonl` contains 15 curated English instruction-response pairs.
- **Response-Only Loss Masking**: Prompts are formatted as `Instruction:\n{instruction}\n\nResponse:\n`. Target tokens for prompt positions are set to `-100`, so the loss is computed **strictly on response tokens**.
- **Results**: Held-out validation instruction loss dropped from `3.3009` (Base Model) to `2.9569` (SFT Model), learning to output response sentences rather than raw narrative prose.
### Training & Checkpoint Experiments
```powershell
# Train Scaled Transformer for 5 epochs with Stride 6 (saves checkpoints to checkpoints_stride_6/)
python train_transformer.py --epochs 5 --step 6 --patience 0 --checkpoint-dir checkpoints_stride_6

### E. Direct Preference Optimization (DPO)
DPO (Rafailov et al., 2023) directly optimizes preference pairs $(x, y_w, y_l)$ using the closed-form objective:
$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = \log\left(1 + \exp\left(-\beta \left[ \left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}\right) - \left(\log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right) \right]\right)\right)$$
- **Analytical Logit Gradients**:
  $$dZ_w = +\beta \sigma(-M) \cdot (p_w - \mathbf{1}_{y_w}) / N_w$$
  $$dZ_l = -\beta \sigma(-M) \cdot (p_l - \mathbf{1}_{y_l}) / N_l$$
- **Results**: DPO loss dropped from $0.6937 \approx \ln(2)$ to $0.4323$, the preference margin increased from $-0.0010$ to $+0.6170$, and preference accuracy reached 100%.
# Train with Stride 2 for rapid fitting
python train_transformer.py --epochs 5 --step 2 --patience 0 --checkpoint-dir checkpoints_stride_2
```

---
### Evaluation & Automated Test Suite
```powershell
# Run the complete 39-test unit and regression verification suite
python test_transformer.py

## 4. Stage 8 Benchmark Results Across Checkpoints
# Run comprehensive reliability, perplexity, diversity, and safety audit
python evaluate_model.py
```

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
### Regression Verification for Earlier Stages
```powershell
# Stage 1 & 2: Student score predictions
python predict.py --hours 8.0 --questions 50

> [!IMPORTANT]
> **Understanding Small Model Limitations & Safety**:
> Commercial models (GPT-4, Claude) possess billions of parameters and train on trillions of tokens. Our model is built from scratch with ~25,000 parameters and trained on 29,370 characters (~5,000 words).
> - It learns real English subwords, basic syllable transitions, and format alignment.
> - However, it **does not possess factual world knowledge, reasoning, or human-level safety understanding**.
> - In the safety audit on sensitive probes, it generates pseudo-English prose from its learned distribution. Safety in language models requires immense scale, external verifiers, and extensive alignment—it cannot be faked with small prompts.
# Stage 3: Character tokenization round-trip demo
python demo_tokenizer.py

---
# Stage 4: Bigram generation
python generate.py --prompt "The " --length 30

## 5. Commands to Run Stage 8

### 1. Compile and Audit Expanded Corpus
```powershell
python prepare_dataset.py
# Stage 7: BPE subword segmentation inspector
python demo_bpe.py
```

### 2. Run the Full 36-Test Suite
```powershell
python test_transformer.py
```
---

### 3. Train the Zero-Leakage Base Transformer
```powershell
python train_transformer.py --tokenizer bpe --num-merges 80 --epochs 35 --patience 6
```
## 🧪 9. Automated Test Suite (All 39 Tests Passing)

### 4. Run Supervised Fine-Tuning (SFT)
```powershell
python train_sft.py --epochs 20 --lr 0.001
```
Running `python test_transformer.py` verifies the entire system from linear algebra to direct preference optimization:

### 5. Run Direct Preference Optimization (DPO)
```powershell
python train_dpo.py --epochs 15 --beta 0.1 --lr 0.0005
```
```text
======================================================================
      STAGE 8+ COMPREHENSIVE TEST SUITE (CORRECTNESS, SFT & DPO)
======================================================================
Test 1: Tensor and array shapes throughout model...               [PASS]
Test 2: Stable softmax and probability sums...                    [PASS]
Test 3: Causal masking check...                                   [PASS]
Test 4: Future-token leakage test...                              [PASS]
Test 5: Finite cross-entropy loss...                              [PASS]
Test 6: Gradient shapes and finite values...                      [PASS]
Test 7: Numerical gradient checking on selected parameters...     [PASS]
Test 8: Loss decrease on small training batch...                  [PASS]
Test 9: Checkpoint save and load fidelity...                      [PASS]
Test 10: Logits consistency before and after reload...            [PASS]
Test 11: Text generation from prompt...                           [PASS]
Test 12: Error handling for invalid prompts and parameters...     [PASS]
Test 13: Regression Check: Stage 1 and Stage 2 models...          [PASS]
Test 14: Regression Check: Stage 3 Tokenizer round-trip...        [PASS]
Test 15: Regression Check: Stage 4 Bigram generator...            [PASS]
Test 16: Sequential data splitting (no future leakage)...         [PASS]
Test 17: L2 weight decay regularizes 2D weights...                [PASS]
Test 18: Early stopping triggers when validation stops...         [PASS]
Test 19: Best checkpoint restoration fidelity...                  [PASS]
Test 20: Dataset preparation normalization & empty-dir guard...   [PASS]
Test 21: Tokenizer vocabulary coverage on clean corpus...         [PASS]
Test 22: Checkpoint vocabulary size compatibility guard...        [PASS]
Test 23: Repetition penalty discourages repeating characters...   [PASS]
Test 24: Top-K sampling restricts sampling candidates...          [PASS]
Test 25: Generation across temperature sweep...                   [PASS]
Test 26: BPE Tokenizer fit, vocabulary growth, and compression... [PASS]
Test 27: BPE Tokenizer round-trip fidelity & <unk> handling...    [PASS]
Test 28: BPE Tokenizer save & load fidelity...                    [PASS]
Test 29: Transformer forward pass and training step with BPE...   [PASS]
Test 30: Text generation with BPETokenizer across sampling...     [PASS]
Test 31: Masked Cross-Entropy Loss with ignore_index=-100...      [PASS]
Test 32: Top-P (Nucleus) Sampling...                              [PASS]
Test 33: End-of-Sequence (EOS) early stopping in generation...    [PASS]
Test 34: Exact token log probabilities extraction...              [PASS]
Test 35: Direct Preference Optimization (DPO) correctness...      [PASS]
Test 36: Training-only tokenizer fitting (zero data leakage)...   [PASS]
Test 37: Leading-space BPE pre-tokenization prevents word fusion. [PASS]
Test 38: Warmup and Cosine Annealing learning rate schedule...    [PASS]
Test 39: Generation formatting and spacing integrity...           [PASS]

### 6. Run Comprehensive Reliability Evaluation
```powershell
python evaluate_model.py
======================================================================
      ALL 39 TESTS PASSED WITH ZERO FAILURES!
======================================================================
```

### 7. Generate Text with Top-P Nucleus Sampling
```powershell
# Generate from Base Model with Top-P = 0.9
python generate_transformer.py --prompt "Once upon a time " --temp 0.6 --top-p 0.9 --length 40
---

# Generate from SFT Model
python generate_transformer.py --checkpoint transformer_sft_checkpoint.npz --prompt "Instruction:\nSummarize the lesson.\n\nResponse:\n" --temp 0.6 --top-p 0.9
```
## 📜 10. Educational Reflections & Principles

### 8. Regression Verification (Stages 1–7)
```powershell
python predict.py --hours 8.0 --questions 50
python generate.py --prompt "The " --length 30
python demo_bpe.py
```
1. **No Black Boxes**: Every forward activation, cross-entropy loss, softmax normalizer, self-attention query-key score, and backpropagation gradient is computed with explicit NumPy matrix multiplication.
2. **Leading-Space Tokenization Matters**: Standard character tokenization creates sequences that are too long for small context windows, while naive subword merging creates word-boundary fusion glitches (e.g. `"timether"`). Adopting GPT-style leading-whitespace pre-tokenization (` ?\w+`) keeps word boundaries clean and invariant to prompt spacing.
3. **Generalization Over Memorization**: A model with 66k parameters trained on 19k tokens will easily memorize the training data if sequence stride is too small. Increasing stride to `step = 6` stabilized training and delivered the lowest validation loss (**3.9075**) and lowest perplexity (**49.77**) in the project.
4. **Honest Evaluation**: We report real training logs, actual evaluation metrics, and unedited model generations without hand-crafted outputs or exaggerated claims.
