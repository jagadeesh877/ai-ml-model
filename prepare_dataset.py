"""
prepare_dataset.py
------------------
Stage 8: Reliable Dataset Preparation & Statistical Analysis.

This script:
1. Scans 'data/raw/' for all .txt files.
2. Carefully normalizes unicode typography (curly quotes, dashes, ellipses) to clean ASCII.
3. Normalizes whitespace (stripping trailing spaces, collapsing redundant blanks, formatting paragraph breaks).
4. Combines the files into a single, clean training corpus: 'data/clean_corpus.txt'.
5. Reports comprehensive dataset statistics: character count, word count, vocabulary diversity,
   paragraph count, and term frequencies.
"""

import os
import glob
import re
from collections import Counter
from typing import List, Tuple, Dict


def normalize_punctuation_and_whitespace(text: str) -> str:
    """
    Carefully normalizes typography, punctuation, and whitespace for language modeling.
    
    Transformations:
    - Smart quotes: “ ” -> "
    - Smart apostrophes: ‘ ’ -> '
    - Dashes: — (em dash), – (en dash) -> -
    - Ellipses: … -> ...
    - Non-breaking spaces: \xa0 -> ' '
    - Line endings: \r\n -> \n
    - Excess whitespace: trailing spaces per line removed, 3+ newlines collapsed to 2.
    """
    # 1. Standardize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Normalize unicode quotation marks and apostrophes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("`", "'")

    # 3. Normalize dashes and ellipses
    text = text.replace("—", " - ").replace("–", " - ")
    text = text.replace("…", "...")

    # 4. Normalize special whitespace
    text = text.replace("\xa0", " ").replace("\t", " ")

    # 5. Clean line-by-line whitespace
    lines = [re.sub(r"[ ]+", " ", line.strip()) for line in text.split("\n")]

    # 6. Recombine and collapse multiple empty lines into double newlines (paragraph breaks)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def build_corpus(
    input_dir: str = "data/raw",
    output_file: str = "data/clean_corpus.txt"
) -> Tuple[str, List[str], int, int]:
    """
    Reads all .txt files from input_dir, normalizes them, and saves to output_file.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Source directory '{input_dir}' does not exist! Please create it and add .txt files.")

    txt_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    if not txt_files:
        raise ValueError(
            f"No .txt files found in '{input_dir}'! Please add clean text files (e.g. fables, stories, essays)."
        )

    corpus_parts = []
    for filepath in txt_files:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_content = f.read()

        cleaned_content = normalize_punctuation_and_whitespace(raw_content)
        if cleaned_content:
            corpus_parts.append(cleaned_content)

    if not corpus_parts:
        raise ValueError(f"All files in '{input_dir}' were empty after cleaning!")

    # Separate distinct stories/chapters by a clean double newline
    combined_text = "\n\n".join(corpus_parts) + "\n"

    # Save to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined_text)

    total_chars = len(combined_text)
    unique_chars = len(set(combined_text))
    return combined_text, txt_files, total_chars, unique_chars


def compute_dataset_statistics(text: str) -> Dict:
    """Computes comprehensive linguistic and statistical metrics for the corpus."""
    words = re.findall(r"\b\w+\b", text)
    word_counts = Counter(w.lower() for w in words)
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    lines = [l for l in text.split("\n") if l.strip()]

    return {
        "characters": len(text),
        "words": len(words),
        "unique_words": len(word_counts),
        "paragraphs": len(paragraphs),
        "lines": len(lines),
        "lexical_diversity": len(word_counts) / max(len(words), 1),
        "top_words": word_counts.most_common(10),
        "people_count": word_counts.get("people", 0)
    }


def main():
    print("=" * 72)
    print("      STAGE 8: RELIABLE DATASET PREPARATION & AUDIT")
    print("=" * 72)

    raw_dir = os.path.join("data", "raw")
    output_path = os.path.join("data", "clean_corpus.txt")

    print(f"Scanning directory: {raw_dir} ...")
    combined_text, files, total_chars, unique_chars = build_corpus(raw_dir, output_path)

    print(f"\n1. Source Files Combined ({len(files)} files):")
    for f in files:
        f_size = os.path.getsize(f)
        print(f"   - {os.path.basename(f):<35} ({f_size:,} bytes)")

    stats = compute_dataset_statistics(combined_text)

    print(f"\n2. Cleaned Corpus Statistics:")
    print(f"   - Output file:          {output_path}")
    print(f"   - Total characters:     {stats['characters']:,}")
    print(f"   - Total words:          {stats['words']:,}")
    print(f"   - Unique words:         {stats['unique_words']:,} (Lexical diversity: {stats['lexical_diversity']:.3f})")
    print(f"   - Unique characters:    {unique_chars}")
    print(f"   - Paragraphs:           {stats['paragraphs']}")
    print(f"   - Lines:                {stats['lines']}")
    print(f"   - Occurrences of 'people': {stats['people_count']}")
    print(f"   - Top 10 words:         {stats['top_words']}")

    print(f"\n3. Corpus Preview (First 300 characters):")
    print("-" * 60)
    print(combined_text[:300].strip() + "\n...")
    print("-" * 60)
    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    main()
