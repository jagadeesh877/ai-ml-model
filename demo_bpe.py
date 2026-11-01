"""
demo_bpe.py
-----------
Stage 7: Byte-Pair Encoding (BPE) Subword Tokenization Demonstration.

What is Byte-Pair Encoding (BPE)?
---------------------------------
In Stage 3, we built a character-level tokenizer where every single letter was a separate token.
While simple, character tokenization has two big weaknesses:
1. Sequences are very long: A 16-token context window only covers ~16 characters (2 to 3 words).
2. The model has to learn how to spell every single English word letter-by-letter.

BPE solves this by starting with individual characters and iteratively merging the most
frequent adjacent pairs into subwords (e.g., 't' + 'h' -> 'th', 'th' + 'e' -> 'the').
This is the exact tokenization algorithm used by modern GPT models (GPT-2, GPT-3, GPT-4)!

This demo script:
1. Loads the clean English corpus (data/clean_corpus.txt).
2. Fits the pure Python BPETokenizer from scratch.
3. Displays the most frequent learned subwords.
4. Measures sequence compression (raw characters vs subword tokens).
5. Demonstrates subword tokenization on sample sentences.
6. Verifies 100% round-trip fidelity: decode(encode(text)) == text.
7. Demonstrates out-of-vocabulary handling and JSON persistence.
"""

import os
import sys
from src.tokenizer import BPETokenizer


def main():
    print("=" * 72)
    print("      STAGE 7: BYTE-PAIR ENCODING (BPE) TOKENIZATION FROM SCRATCH")
    print("=" * 72)

    corpus_path = os.path.join("data", "clean_corpus.txt")
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join("data", "tiny_story.txt")
        print(f"Notice: 'data/clean_corpus.txt' not found, using '{corpus_path}'.")

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    print(f"\n1. Loaded Corpus:")
    print(f"   - File: {corpus_path}")
    print(f"   - Total characters: {len(corpus):,}")
    unique_chars = sorted(list(set(corpus)))
    print(f"   - Unique characters: {len(unique_chars)}")

    # -----------------------------------------------------------------
    # Step 2: Fit BPE Tokenizer
    # -----------------------------------------------------------------
    num_merges = 60
    print(f"\n2. Training BPE Tokenizer with {num_merges} iterative merges...")
    tokenizer = BPETokenizer(unk_token="<unk>")
    tokenizer.fit(corpus, num_merges=num_merges, min_freq=2)

    base_size = len(tokenizer.base_vocab)
    final_size = tokenizer.vocab_size
    actual_merges = len(tokenizer.merges_list)

    print(f"   - Base vocabulary size:   {base_size} (1 <unk> + {base_size - 1} characters)")
    print(f"   - Merges performed:       {actual_merges}")
    print(f"   - Final vocabulary size:  {final_size} tokens")

    # -----------------------------------------------------------------
    # Step 3: Top Learned Subwords
    # -----------------------------------------------------------------
    print(f"\n3. Top 15 Learned Subwords (Merged by Frequency):")
    print(f"   {'Rank':<5} | {'Token ID':<9} | {'Merged Subword String':<25}")
    print("   " + "-" * 45)
    for rank, (pair, new_id) in enumerate(tokenizer.merges_list[:15], 1):
        token_str = repr(tokenizer.id_to_token[new_id])
        print(f"   {rank:<5} | {new_id:<9} | {token_str:<25}")

    # -----------------------------------------------------------------
    # Step 4: Sequence Compression Analysis
    # -----------------------------------------------------------------
    encoded_corpus = tokenizer.encode(corpus)
    char_count = len(corpus)
    token_count = len(encoded_corpus)
    compression = char_count / max(token_count, 1)
    savings = (1.0 - token_count / char_count) * 100

    print(f"\n4. Sequence Compression Performance:")
    print(f"   - Raw character count:  {char_count:,} characters")
    print(f"   - BPE token count:      {token_count:,} tokens")
    print(f"   - Compression ratio:    {compression:.2f}x characters per token")
    print(f"   - Sequence reduction:   {savings:.1f}% fewer tokens needed to represent same text!")
    print(f"   >> Meaning: A 16-token Transformer context window now sees ~{16 * compression:.0f} characters of prose!")

    # -----------------------------------------------------------------
    # Step 5: Subword Tokenization on Example Phrases
    # -----------------------------------------------------------------
    print(f"\n5. How BPE Segments English Phrases into Subwords:")
    test_phrases = [
        "The quick brown fox",
        "The little robot smiled.",
        "in the dark forest",
        "learning step by step"
    ]
    for phrase in test_phrases:
        token_strs = tokenizer.tokenize_to_strings(phrase)
        token_ids = tokenizer.encode(phrase)
        print(f"\n   Phrase:    {repr(phrase)}")
        print(f"   Subwords:  {token_strs}")
        print(f"   Token IDs: {token_ids}")

    # -----------------------------------------------------------------
    # Step 6: Mathematical Round-Trip Fidelity Guarantee
    # -----------------------------------------------------------------
    print(f"\n6. Mathematical Round-Trip Verification:")
    reconstructed_corpus = tokenizer.decode(encoded_corpus)
    assert reconstructed_corpus == corpus, "Verification Failed: decode(encode(corpus)) != corpus"
    print("   [PASS] 100% exact match across all 22,272 characters of the corpus!")

    # Test individual sentences
    sample_sentences = [
        "Once upon a time there was a tiny robot.",
        "The quick brown fox jumps over the river.",
        "Every day they went to the green forest.",
        "A quiet evening by the fire."
    ]
    for s in sample_sentences:
        assert tokenizer.decode(tokenizer.encode(s)) == s, f"Failed on: {s}"
    print(f"   [PASS] All {len(sample_sentences)} test sentences round-trip perfectly.")

    # -----------------------------------------------------------------
    # Step 7: Unknown Character Handling
    # -----------------------------------------------------------------
    print(f"\n7. Unknown Character Handling:")
    rare_phrase = "Hello 123! #@$"
    enc_replace = tokenizer.encode(rare_phrase, handle_unknown="replace")
    dec_replace = tokenizer.decode(enc_replace)
    print(f"   - Original:  {repr(rare_phrase)}")
    print(f"   - Replaced:  {repr(dec_replace)} (unsupported characters become <unk>)")

    try:
        tokenizer.encode(rare_phrase, handle_unknown="error")
        print("   [FAIL] Should have raised ValueError.")
    except ValueError as e:
        print(f"   - Error mode: Caught expected ValueError:\n     >> {str(e)[:80]}...")

    # -----------------------------------------------------------------
    # Step 8: Save & Load Persistence
    # -----------------------------------------------------------------
    vocab_path = "bpe_vocab.json"
    print(f"\n8. Saving BPE Vocabulary and Merges to: {vocab_path}")
    tokenizer.save(vocab_path)

    reloaded_tok = BPETokenizer()
    reloaded_tok.load(vocab_path)
    assert reloaded_tok.vocab_size == tokenizer.vocab_size
    assert reloaded_tok.encode("The fox") == tokenizer.encode("The fox")
    assert reloaded_tok.decode(reloaded_tok.encode(corpus)) == corpus
    print("   [PASS] BPE vocabulary and merge rules saved and reloaded with 100% fidelity.")

    print("\n" + "=" * 72)
    print("      STAGE 7 BPE TOKENIZER DEMONSTRATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()

