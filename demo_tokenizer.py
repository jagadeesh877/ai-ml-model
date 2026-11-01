"""
demo_tokenizer.py
-----------------
Demonstrates and verifies Stage 3: Tokenization & Text Representation.

This script:
1. Loads our sample text corpus (data/tiny_story.txt).
2. Fits the CharTokenizer from scratch.
3. Tests encoding (text -> token IDs) and decoding (token IDs -> text).
4. Formally verifies that decode(encode(text)) == text (Round-Trip Guarantee).
5. Demonstrates error handling for unknown characters and out-of-bounds IDs.
6. Prepares input/target sequences (X -> y) showing how GPT models train on text.
"""

import os
from src.tokenizer import CharTokenizer, create_language_modeling_sequences


def main():
    print("=" * 70)
    print("      STAGE 3: TOKENIZATION & TEXT REPRESENTATION FROM SCRATCH")
    print("=" * 70)

    corpus_path = os.path.join("data", "tiny_story.txt")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file '{corpus_path}' not found!")

    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"\n1. Loaded text corpus from: {corpus_path}")
    print(f"   Total characters in corpus: {len(raw_text)}")
    print(f"   Sample text:\n   ---\n   {raw_text.strip()[:140]}...\n   ---")

    # -------------------------------------------------------------
    # Step 2: Fit Tokenizer
    # -------------------------------------------------------------
    tokenizer = CharTokenizer()
    tokenizer.fit(raw_text)

    print(f"\n2. Fitted Tokenizer Vocabulary:")
    print(f"   Vocabulary Size (including <unk>): {tokenizer.vocab_size} unique tokens")
    print(f"   Characters in Vocabulary: {repr(''.join(tokenizer.vocab[1:]))}")

    # -------------------------------------------------------------
    # Step 3: Encoding & Decoding
    # -------------------------------------------------------------
    sample_phrase = "The robot smiled."
    encoded_ids = tokenizer.encode(sample_phrase)
    decoded_text = tokenizer.decode(encoded_ids)

    print(f"\n3. Encoding & Decoding Demonstration:")
    print(f"   Original phrase : '{sample_phrase}'")
    print(f"   Encoded IDs     : {encoded_ids}")
    print(f"   Decoded phrase  : '{decoded_text}'")

    # -------------------------------------------------------------
    # Step 4: Round-Trip Verification (Mathematical Guarantee)
    # -------------------------------------------------------------
    print(f"\n4. Verifying Round-Trip Fidelity:")
    # Test on entire corpus
    full_encoded = tokenizer.encode(raw_text)
    full_decoded = tokenizer.decode(full_encoded)
    assert full_decoded == raw_text, "Verification Failed: Decoded text does not match original corpus!"
    print("   [PASS] decode(encode(corpus)) == corpus (100% exact match across all characters!)")

    test_sentences = [
        "How do you learn to fly?",
        "Every journey begins with a single step.",
        "Practice every day, learn from your errors.",
        "stars."
    ]
    for sentence in test_sentences:
        reconstructed = tokenizer.decode(tokenizer.encode(sentence))
        assert reconstructed == sentence, f"Mismatch on sentence: {sentence}"
    print(f"   [PASS] Tested {len(test_sentences)} individual sentences: all reconstructed perfectly.")

    # -------------------------------------------------------------
    # Step 5: Error Handling Demonstration
    # -------------------------------------------------------------
    print(f"\n5. Error Handling Demonstration:")
    foreign_phrase = "Hello world! 123 @#"  # Contains '!', '1', '2', '3', '@', '#' not in story vocab

    # 5a. Replace mode: replaces unsupported characters with <unk>
    encoded_replaced = tokenizer.encode(foreign_phrase, handle_unknown="replace")
    decoded_replaced = tokenizer.decode(encoded_replaced)
    print(f"   5a. Replace Mode (handle_unknown='replace'):")
    print(f"       Input   : '{foreign_phrase}'")
    print(f"       Encoded : {encoded_replaced}")
    print(f"       Decoded : '{decoded_replaced}'")

    # 5b. Error mode: raises clear exception with supported characters list
    print(f"   5b. Error Mode (handle_unknown='error'):")
    try:
        tokenizer.encode(foreign_phrase, handle_unknown="error")
        print("       [FAIL] Error should have been raised but was not.")
    except ValueError as e:
        print(f"       [PASS] Caught expected ValueError:\n       >> {str(e)[:100]}...")

    # 5c. Invalid token ID check
    print(f"   5c. Out-of-bounds Token ID Check:")
    try:
        tokenizer.decode([9999])
        print("       [FAIL] Out of bounds ID should have raised ValueError.")
    except ValueError as e:
        print(f"       [PASS] Caught expected ValueError:\n       >> {e}")

    # -------------------------------------------------------------
    # Step 6: Dataset Preparation for Next-Character Prediction (GPT Style)
    # -------------------------------------------------------------
    print(f"\n6. Language Model Sequence Preparation (Context -> Next Token):")
    context_length = 6
    X_seqs, y_targets = create_language_modeling_sequences(full_encoded, context_length=context_length)
    print(f"   Generated {len(X_seqs)} training pairs using context_length={context_length}")

    print("\n   Sample (Context -> Next Character) Training Pairs:")
    print("   " + "-" * 62)
    print(f"   {'Input Context (X)':<22} | {'Context IDs':<18} | {'Target (y)':<12}")
    print("   " + "-" * 62)
    for i in range(8):
        ctx_text = repr(tokenizer.decode(X_seqs[i]))
        ctx_ids = str(X_seqs[i])
        tgt_char = repr(tokenizer.decode([y_targets[i]]))
        print(f"   {ctx_text:<22} | {ctx_ids:<18} | {tgt_char:<12}")
    print("   " + "-" * 62)

    # -------------------------------------------------------------
    # Step 7: Save Vocabulary Checkpoint
    # -------------------------------------------------------------
    vocab_file = "tokenizer_vocab.json"
    tokenizer.save(vocab_file)
    print(f"\n7. Saved tokenizer vocabulary to: {vocab_file}")
    print("=" * 70)
    print("      STAGE 3 VERIFICATION COMPLETE: ALL CHECKS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
