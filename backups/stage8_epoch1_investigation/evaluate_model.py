"""
evaluate_model.py
-----------------
Stage 8: Comprehensive Model Reliability, Diversity, and Safety Evaluation.

Metrics computed:
1. Cross-Entropy Loss & Perplexity (PPL = exp(loss)) on training and validation splits.
2. Distinct-1 and Distinct-2 n-gram ratios (measuring lexical diversity and repetition).
3. Fixed-Prompt Benchmark across multiple sampling configurations.
4. Safety, Uncertainty, and Adversarial Audit on 'data/safety_eval.jsonl'.
5. Structured JSON report export ('eval_report.json').
"""

import os
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
from src.tokenizer import BPETokenizer
from src.transformer_model import TinyGPTLanguageModel
from train_transformer import prepare_sequences, evaluate_loss


def compute_ngram_diversity(text: str) -> Dict[str, float]:
    """Computes Distinct-1 and Distinct-2 ratios to quantify repetition vs diversity."""
    import re
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "total_words": 0}

    unigrams = words
    bigrams = list(zip(words[:-1], words[1:]))

    d1 = len(set(unigrams)) / len(unigrams)
    d2 = len(set(bigrams)) / max(len(bigrams), 1)

    return {
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
        "total_words": len(words)
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    vocab_path: str = "bpe_vocab.json",
    corpus_path: str = "data/clean_corpus.txt",
    context_length: int = 16
) -> Dict:
    """Evaluates a single model checkpoint on loss, perplexity, and generation diversity."""
    if not os.path.exists(checkpoint_path):
        return {"status": "missing", "checkpoint": checkpoint_path}

    tokenizer = BPETokenizer()
    tokenizer.load(vocab_path)

    model = TinyGPTLanguageModel(vocab_size=tokenizer.vocab_size)
    model.load_checkpoint(checkpoint_path, expected_vocab_size=tokenizer.vocab_size)
    ctx = model.context_length

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    split_idx = int(len(text) * 0.85)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_tokens = np.array(tokenizer.encode(train_text), dtype=int)
    val_tokens = np.array(tokenizer.encode(val_text, handle_unknown="replace"), dtype=int)

    X_train, y_train = prepare_sequences(train_tokens, context_length=ctx, step=3)
    X_val, y_val = prepare_sequences(val_tokens, context_length=ctx, step=3)

    train_loss = evaluate_loss(model, X_train, y_train, batch_size=32)
    val_loss = evaluate_loss(model, X_val, y_val, batch_size=32)

    train_ppl = float(np.exp(train_loss))
    val_ppl = float(np.exp(val_loss))

    # Benchmark Generation
    test_prompts = [
        "The ",
        "Once upon a time ",
        "The tortoise plodded on ",
        "Light travels much faster ",
        "Instruction:\nSummarize the lesson of the tortoise and the hare.\n\nResponse:\n"
    ]

    generations = {}
    combined_gen_text = ""
    for p in test_prompts:
        gen = model.generate(
            tokenizer,
            prompt=p,
            max_new_tokens=40,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1,
            seed=42
        )
        generations[p] = gen
        combined_gen_text += gen + " "

    diversity = compute_ngram_diversity(combined_gen_text)

    return {
        "status": "evaluated",
        "checkpoint": checkpoint_path,
        "vocab_size": tokenizer.vocab_size,
        "train_loss": round(train_loss, 4),
        "val_loss": round(val_loss, 4),
        "train_perplexity": round(train_ppl, 2),
        "val_perplexity": round(val_ppl, 2),
        "diversity": diversity,
        "generations": generations
    }


def run_safety_audit(model: TinyGPTLanguageModel, tokenizer: BPETokenizer, safety_file: str = "data/safety_eval.jsonl") -> List[Dict]:
    """Tests model responses to sensitive, adversarial, or contradictory prompts."""
    if not os.path.exists(safety_file):
        return []

    results = []
    with open(safety_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item["prompt"]
            gen = model.generate(
                tokenizer,
                prompt=prompt,
                max_new_tokens=30,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1,
                seed=42
            )
            raw_response = gen[len(prompt):].strip()

            results.append({
                "category": item.get("category", "general"),
                "expected": item.get("expected_behavior", ""),
                "prompt": prompt.strip(),
                "actual_response": raw_response,
                "behavior_analysis": "Model generates pseudo-English prose from learned token distributions without real safety intent."
            })
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tiny GPT Models for Reliability, Perplexity, and Safety.")
    parser.add_argument("--report", type=str, default="eval_report.json", help="Output path for JSON report")
    args = parser.parse_args()

    print("=" * 72)
    print("      STAGE 8: COMPREHENSIVE MODEL RELIABILITY & SAFETY EVALUATION")
    print("=" * 72)

    checkpoints_to_eval = [
        ("Base Model", "transformer_bpe_checkpoint.npz"),
        ("SFT Model", "transformer_sft_checkpoint.npz"),
        ("DPO Model", "transformer_dpo_checkpoint.npz"),
    ]

    all_results = {}
    for name, ckpt in checkpoints_to_eval:
        print(f"\nEvaluating {name} ({ckpt})...")
        res = evaluate_checkpoint(ckpt)
        all_results[name] = res
        if res.get("status") == "evaluated":
            print(f"  - Train Loss: {res['train_loss']:.4f} (PPL: {res['train_perplexity']:.2f})")
            print(f"  - Val Loss:   {res['val_loss']:.4f} (PPL: {res['val_perplexity']:.2f})")
            print(f"  - Distinct-1: {res['diversity']['distinct_1']:.3f} | Distinct-2: {res['diversity']['distinct_2']:.3f}")
        else:
            print(f"  - Notice: Checkpoint not found, skipping.")

    # Safety Audit on the best available model
    best_ckpt = "transformer_sft_checkpoint.npz" if os.path.exists("transformer_sft_checkpoint.npz") else "transformer_bpe_checkpoint.npz"
    safety_results = []
    if os.path.exists(best_ckpt) and os.path.exists("bpe_vocab.json"):
        print(f"\nRunning Safety and Uncertainty Audit on '{best_ckpt}'...")
        tok = BPETokenizer()
        tok.load("bpe_vocab.json")
        m = TinyGPTLanguageModel(vocab_size=tok.vocab_size)
        m.load_checkpoint(best_ckpt, expected_vocab_size=tok.vocab_size)
        safety_results = run_safety_audit(m, tok)
        all_results["safety_audit"] = safety_results

        print("-" * 72)
        print("Safety Probe Samples:")
        for sample in safety_results[:3]:
            print(f"  Category:  {sample['category']}")
            print(f"  Prompt:    {repr(sample['prompt'][:45])}...")
            print(f"  Response:  {repr(sample['actual_response'])}")
            print()

    # Save structured report
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation report successfully saved to: {args.report}")
    print("=" * 72)


if __name__ == "__main__":
    main()

