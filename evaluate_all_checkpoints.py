import os
import json
import re
import numpy as np
from src.tokenizer import BPETokenizer
from src.transformer_model import TinyGPTLanguageModel
from train_transformer import prepare_sequences, evaluate_loss

# Prompts
prompts = [
    "once upon time",
    "The brave knight ",
    "In the middle of the garden ",
    "A clever fox noticed ",
    "Long ago in a distant kingdom "
]

tok = BPETokenizer()
tok.load("bpe_vocab.json")

with open("data/clean_corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

split_idx = int(len(text) * 0.85)
train_text = text[:split_idx]
val_text = text[split_idx:]

train_toks = np.array(tok.encode(train_text))
val_toks = np.array(tok.encode(val_text, handle_unknown="replace"))

# Common English dictionary for word formation measurement
import urllib.request
# We extract all words from train and val as the reference dictionary + common English
corpus_words = set(re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()))

def analyze_generations(gen_dict):
    total_words = 0
    real_words = 0
    all_words = []
    for prompt, gen in gen_dict.items():
        # Only evaluate the continuation, after prompt
        continuation = gen[len(prompt):]
        words = re.findall(r"\b[a-zA-Z]{2,}\b", continuation.lower())
        all_words.extend(words)
        for w in words:
            total_words += 1
            if w in corpus_words:
                real_words += 1
    
    word_acc = (real_words / max(total_words, 1)) * 100.0
    
    # Distinct n-grams
    d1 = len(set(all_words)) / max(len(all_words), 1)
    bigrams = list(zip(all_words[:-1], all_words[1:]))
    d2 = len(set(bigrams)) / max(len(bigrams), 1)
    
    return {
        "word_formation_pct": round(word_acc, 1),
        "distinct_1": round(d1, 3),
        "distinct_2": round(d2, 3),
        "total_words": total_words
    }

results = {}

for run_name, ckpt_dir, step in [("Stride 6", "checkpoints_stride_6", 6), ("Stride 2", "checkpoints_stride_2", 2)]:
    results[run_name] = {}
    X_tr, y_tr = prepare_sequences(train_toks, context_length=32, step=step)
    X_v, y_v = prepare_sequences(val_toks, context_length=32, step=step)
    
    for ep in range(1, 6):
        ckpt_file = os.path.join(ckpt_dir, f"checkpoint_epoch_{ep}.npz")
        if not os.path.exists(ckpt_file):
            continue
            
        m = TinyGPTLanguageModel(vocab_size=tok.vocab_size)
        m.load_checkpoint(ckpt_file, expected_vocab_size=tok.vocab_size)
        
        tr_loss = evaluate_loss(m, X_tr, y_tr, batch_size=32)
        v_loss = evaluate_loss(m, X_v, y_v, batch_size=32)
        
        gen_outputs = {}
        for p in prompts:
            g = m.generate(tok, prompt=p, max_new_tokens=40, temperature=0.6, top_p=0.9, repetition_penalty=1.1, seed=42)
            gen_outputs[p] = g
            
        metrics = analyze_generations(gen_outputs)
        
        results[run_name][f"Epoch {ep}"] = {
            "train_loss": round(tr_loss, 4),
            "train_ppl": round(np.exp(tr_loss), 2),
            "val_loss": round(v_loss, 4),
            "val_ppl": round(np.exp(v_loss), 2),
            "word_formation_pct": metrics["word_formation_pct"],
            "distinct_1": metrics["distinct_1"],
            "distinct_2": metrics["distinct_2"],
            "generations": gen_outputs
        }

with open("checkpoint_evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Evaluation across all 10 checkpoints completed successfully!")

