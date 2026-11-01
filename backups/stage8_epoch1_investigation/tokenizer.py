"""
tokenizer.py
------------
Stage 3: Tokenization & Text Representation (Pure Python).

Why does an AI / GPT model need a Tokenizer?
Computers cannot understand human letters or words directly. All machine learning
models (including GPT-4 and ChatGPT) only understand numbers, vectors, and matrices.

A Tokenizer acts as a two-way dictionary:
1. Encoding:  Turns human text into numeric IDs: "cat" -> [3, 1, 20]
2. Decoding:  Turns numeric IDs back into human text: [3, 1, 20] -> "cat"

In this module, we build a Character-Level Tokenizer completely from scratch
using standard Python (no external dependencies, no PyTorch).
"""

import json
from typing import List, Tuple, Dict
import numpy as np


class CharTokenizer:
    """
    A character-level tokenizer that maps every unique character to a unique integer ID.
    """
    def __init__(self, unk_token: str = "<unk>"):
        """
        Initializes the tokenizer.
        
        Args:
            unk_token (str): Special token representing unknown characters not seen in the vocabulary.
        """
        self.unk_token = unk_token
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab: List[str] = []
        self.vocab_size: int = 0
        self.is_fitted: bool = False

    def fit(self, text: str):
        """
        Learns the vocabulary from a body of text.
        Finds all unique characters, sorts them, and assigns each a numeric ID.
        """
        if not text:
            raise ValueError("Cannot fit tokenizer on empty text.")

        # Extract unique characters and sort them for consistent, deterministic IDs
        unique_chars = sorted(list(set(text)))

        # Always include the unknown token at index 0
        self.vocab = [self.unk_token] + unique_chars
        self.vocab_size = len(self.vocab)

        # Build bidirectional lookup tables
        self.char_to_id = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.id_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}
        self.is_fitted = True

    def encode(self, text: str, handle_unknown: str = "replace") -> List[int]:
        """
        Converts a string of text into a list of integer token IDs.
        
        Args:
            text (str): Input text to encode.
            handle_unknown (str):
                - "replace": Replaces unknown characters with the <unk> token ID.
                - "error": Raises a ValueError if an unknown character is encountered.
                
        Returns:
            List[int]: The list of integer token IDs.
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer has not been fitted on any text yet. Call fit() first.")

        token_ids: List[int] = []
        unk_id = self.char_to_id[self.unk_token]

        for char in text:
            if char in self.char_to_id:
                token_ids.append(self.char_to_id[char])
            else:
                if handle_unknown == "replace":
                    token_ids.append(unk_id)
                elif handle_unknown == "error":
                    raise ValueError(
                        f"Character '{char}' (Unicode: U+{ord(char):04X}) is not in the vocabulary! "
                        f"Supported characters are: {sorted([c for c in self.char_to_id.keys() if c != self.unk_token])}"
                    )
                else:
                    raise ValueError(f"Unknown handle_unknown option: '{handle_unknown}'. Use 'replace' or 'error'.")

        return token_ids

    def decode(self, token_ids: List[int], ignore_unk: bool = False) -> str:
        """
        Converts a list of integer token IDs back into human-readable text.
        
        Args:
            token_ids (List[int]): The list of numeric token IDs.
            ignore_unk (bool): If True, skips <unk> tokens in the decoded output.
            
        Returns:
            str: The reconstructed text string.
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer has not been fitted on any text yet. Call fit() first.")

        chars: List[str] = []
        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError(f"Expected integer token ID, got {type(token_id).__name__}: {token_id}")

            if token_id in self.id_to_char:
                char = self.id_to_char[token_id]
                if char == self.unk_token and ignore_unk:
                    continue
                chars.append(char)
            else:
                raise ValueError(
                    f"Token ID {token_id} is out of vocabulary bounds! Valid range is 0 to {self.vocab_size - 1}."
                )

        return "".join(chars)

    def save(self, filepath: str):
        """Saves the vocabulary to a JSON file."""
        data = {
            "unk_token": self.unk_token,
            "vocab": self.vocab,
            "vocab_size": self.vocab_size
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str):
        """Loads a vocabulary from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.unk_token = data["unk_token"]
        self.vocab = data["vocab"]
        self.vocab_size = data["vocab_size"]
        self.char_to_id = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.id_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}
        self.is_fitted = True


def create_language_modeling_sequences(
    token_ids: List[int],
    context_length: int = 4,
    step: int = 1
) -> Tuple[List[List[int]], List[int]]:
    """
    Creates input contexts (X) and next-token targets (y) for training a language model.
    
    How GPT models learn:
    They learn by predicting the *very next token* given the preceding context!
    
    Example:
        Suppose the token IDs represent the letters: ['T', 'h', 'e', ' ', 'c', 'a', 't']
        With context_length = 3:
            Context (X)            Target (y)
            ['T', 'h', 'e']   ->   ' '
            ['h', 'e', ' ']   ->   'c'
            ['e', ' ', 'c']   ->   'a'
            [' ', 'c', 'a']   ->   't'
            
    Args:
        token_ids (List[int]): The full stream of token IDs.
        context_length (int): How many characters the model looks at to guess the next one.
        step (int): Stride between consecutive windows (usually 1).
        
    Returns:
        X (List[List[int]]): List of context sequences, each of length `context_length`.
        y (List[int]): List of next-token targets.
    """
    if len(token_ids) <= context_length:
        raise ValueError(
            f"Token list length ({len(token_ids)}) must be greater than context_length ({context_length})."
        )

    X: List[List[int]] = []
    y: List[int] = []

    for i in range(0, len(token_ids) - context_length, step):
        context = token_ids[i : i + context_length]
        target = token_ids[i + context_length]
        X.append(context)
        y.append(target)

    return X, y


class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Subword Tokenizer from Scratch (Pure Python).
    
    Why Byte-Pair Encoding (BPE)?
    - Character-level tokenization (Stage 3):
        * Vocabulary is small (~60 characters).
        * But sequences are very long! A 16-token context only covers 16 characters (~2 words).
        * The model has to learn spelling from scratch letter by letter.
    - Word-level tokenization:
        * Vocabulary is gigantic (50,000+ words).
        * Cannot handle any word not in the dictionary (Out-of-Vocabulary problem).
    - Subword tokenization via BPE with Leading-Whitespace Pre-tokenization:
        * Best of both worlds!
        * Pre-tokenization groups words with their optional leading space (GPT-style).
        * Merges are learned strictly within word/punctuation chunks, never across word boundaries.
        * Prevents word-boundary fusion bugs (e.g. "time" + "there" -> "timether").
        * Frequent words (" the", " and", " was", " in", " of", " to", " time") become single tokens!
        * 100% lossless round-trip guarantee: decode(encode(text)) == text.
    """
    def __init__(self, unk_token: str = "<unk>"):
        """
        Initializes the BPE tokenizer.
        
        Args:
            unk_token (str): Special token representing unknown characters not seen during training.
        """
        import re
        self.unk_token = unk_token
        # Universal unicode regex: letters with optional leading space, digits, punctuation/symbols, whitespace, fallback
        self.pat = re.compile(r" ?[^\s\W0-9_]+| ?[0-9]+| ?[^\s\w]+|\s+|.")
        self.vocab: List[str] = []
        self.vocab_size: int = 0
        self.base_vocab: List[str] = []
        self.char_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.id_to_char: Dict[int, str] = {}  # Compatibility alias
        self.merges_list: List[Tuple[Tuple[int, int], int]] = []
        self.merges: Dict[Tuple[int, int], Tuple[int, int]] = {}  # pair -> (new_id, rank)
        self.is_fitted: bool = False

    def fit(self, text: str, num_merges: int = 60, min_freq: int = 2):
        """
        Learns BPE subword vocabulary and merge rules from a training corpus.
        
        Algorithm:
        1. Base vocabulary: Extract all unique characters and prepend <unk>.
        2. Pre-tokenize text into word and whitespace chunks using regex.
        3. Convert chunks into lists of base character IDs.
        4. Repeat up to `num_merges` times:
           a. Count frequencies of all adjacent pairs (id_a, id_b) within chunks.
           b. Pick the pair with highest frequency (deterministic tie-breaking).
           c. If max frequency < min_freq, stop early.
           d. Allocate a new token ID for this merged pair.
           e. Append merged string to vocabulary and record the merge rule.
           f. Replace all occurrences of [id_a, id_b] within each chunk with new_id.
           
        Args:
            text (str): Training text corpus.
            num_merges (int): Number of subword merge operations to learn.
            min_freq (int): Minimum frequency required to perform a merge.
        """
        if not text:
            raise ValueError("Cannot fit BPETokenizer on empty text.")

        from collections import Counter

        # 1. Base vocabulary from unique characters
        unique_chars = sorted(list(set(text)))
        self.base_vocab = [self.unk_token] + unique_chars
        self.vocab = list(self.base_vocab)
        self.char_to_id = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.id_to_token = {idx: ch for idx, ch in enumerate(self.vocab)}
        self.id_to_char = self.id_to_token

        # 2. Pre-tokenize into chunks so merges do not cross word boundaries
        chunks = self.pat.findall(text)
        chunk_tokens: List[List[int]] = [
            [self.char_to_id[ch] for ch in chunk]
            for chunk in chunks
        ]

        self.merges_list = []
        self.merges = {}

        # 3. Iterative pair merging within chunks
        for rank in range(num_merges):
            pair_counts = Counter()
            for c_tokens in chunk_tokens:
                if len(c_tokens) >= 2:
                    for p in zip(c_tokens[:-1], c_tokens[1:]):
                        pair_counts[p] += 1

            if not pair_counts:
                break

            # Pick pair with highest frequency. Deterministic tie-breaking.
            best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], -p[0], -p[1]))
            if pair_counts[best_pair] < min_freq:
                break

            new_id = len(self.vocab)
            new_token_str = self.id_to_token[best_pair[0]] + self.id_to_token[best_pair[1]]

            self.vocab.append(new_token_str)
            self.id_to_token[new_id] = new_token_str
            self.merges_list.append((best_pair, new_id))
            self.merges[best_pair] = (new_id, rank)

            # Replace occurrences of best_pair in each chunk
            new_chunk_tokens: List[List[int]] = []
            for c_tokens in chunk_tokens:
                if len(c_tokens) < 2:
                    new_chunk_tokens.append(c_tokens)
                    continue
                new_c: List[int] = []
                j = 0
                n = len(c_tokens)
                while j < n:
                    if j < n - 1 and (c_tokens[j], c_tokens[j + 1]) == best_pair:
                        new_c.append(new_id)
                        j += 2
                    else:
                        new_c.append(c_tokens[j])
                        j += 1
                new_chunk_tokens.append(new_c)
            chunk_tokens = new_chunk_tokens

        self.vocab_size = len(self.vocab)
        self.is_fitted = True

    def encode(self, text: str, handle_unknown: str = "replace") -> List[int]:
        """
        Encodes human text into BPE token IDs with leading whitespace pre-tokenization.
        
        Steps:
        1. Splits text into word/punctuation/whitespace chunks.
        2. Maps each character to its initial base character ID.
        3. Applies the learned merge rules in the exact order they were learned.
        
        Args:
            text (str): Input text string.
            handle_unknown (str): 'replace' (replaces unknown chars with <unk>)
                                 or 'error' (raises ValueError).
        Returns:
            List[int]: List of BPE token IDs.
        """
        if not self.is_fitted:
            raise RuntimeError("BPETokenizer has not been fitted yet. Call fit() or load() first.")

        if not text:
            return []

        unk_id = self.char_to_id[self.unk_token]
        chunks = self.pat.findall(text)
        all_tokens: List[int] = []

        for chunk in chunks:
            c_tokens: List[int] = []
            for char in chunk:
                if char in self.char_to_id:
                    c_tokens.append(self.char_to_id[char])
                else:
                    if handle_unknown == "replace":
                        c_tokens.append(unk_id)
                    elif handle_unknown == "error":
                        raise ValueError(
                            f"Character {repr(char)} (Unicode U+{ord(char):04X}) is not in the base vocabulary!"
                        )
                    else:
                        raise ValueError(f"Unknown handle_unknown option: '{handle_unknown}'. Use 'replace' or 'error'.")

            for pair, new_id in self.merges_list:
                if len(c_tokens) < 2:
                    break
                new_c: List[int] = []
                j = 0
                n = len(c_tokens)
                while j < n:
                    if j < n - 1 and (c_tokens[j], c_tokens[j + 1]) == pair:
                        new_c.append(new_id)
                        j += 2
                    else:
                        new_c.append(c_tokens[j])
                        j += 1
                c_tokens = new_c

            all_tokens.extend(c_tokens)

        return all_tokens

    def decode(self, token_ids: List[int], ignore_unk: bool = False) -> str:
        """
        Decodes a sequence of BPE token IDs back into human-readable text.
        
        Args:
            token_ids (List[int]): List of integer token IDs.
            ignore_unk (bool): If True, omits <unk> tokens.
            
        Returns:
            str: Reconstructed text string.
        """
        if not self.is_fitted:
            raise RuntimeError("BPETokenizer has not been fitted yet. Call fit() or load() first.")

        pieces: List[str] = []
        for tid in token_ids:
            if not isinstance(tid, (int, np.integer)):
                raise TypeError(f"Expected integer token ID, got {type(tid).__name__}: {tid}")
            tid = int(tid)
            if tid < 0 or tid >= self.vocab_size:
                raise ValueError(
                    f"Token ID {tid} is out of bounds! Valid range is 0 to {self.vocab_size - 1}."
                )
            token_str = self.id_to_token[tid]
            if token_str == self.unk_token and ignore_unk:
                continue
            pieces.append(token_str)

        return "".join(pieces)

    def tokenize_to_strings(self, text: str) -> List[str]:
        """
        Encodes text and returns the corresponding subword token strings.
        Helpful for inspecting how BPE segments words into subwords.
        """
        encoded_ids = self.encode(text)
        return [self.id_to_token[tid] for tid in encoded_ids]

    def get_compression_ratio(self, text: str) -> float:
        """
        Calculates the compression ratio = len(raw text characters) / len(BPE tokens).
        A ratio of 1.6 means the BPE tokens represent 1.6 characters on average.
        """
        if not text:
            return 1.0
        encoded = self.encode(text)
        return len(text) / max(len(encoded), 1)

    def save(self, filepath: str):
        """
        Saves the BPE vocabulary and merge rules to a JSON file.
        """
        data = {
            "tokenizer_type": "bpe",
            "unk_token": self.unk_token,
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "base_vocab": self.base_vocab,
            "base_vocab_size": len(self.base_vocab),
            "num_merges": len(self.merges_list),
            "merges": [
                [[int(pair[0]), int(pair[1])], int(new_id)]
                for pair, new_id in self.merges_list
            ]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str):
        """
        Loads BPE vocabulary and merge rules from a JSON file.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.unk_token = data.get("unk_token", "<unk>")
        self.vocab = data["vocab"]
        self.vocab_size = data["vocab_size"]
        self.base_vocab = data.get("base_vocab", [])
        if not self.base_vocab:
            num_m = data.get("num_merges", len(data.get("merges", [])))
            self.base_vocab = self.vocab[: len(self.vocab) - num_m]

        self.char_to_id = {ch: idx for idx, ch in enumerate(self.base_vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        self.id_to_char = self.id_to_token

        self.merges_list = [
            ((int(item[0][0]), int(item[0][1])), int(item[1]))
            for item in data.get("merges", [])
        ]
        self.merges = {
            pair: (new_id, rank)
            for rank, (pair, new_id) in enumerate(self.merges_list)
        }
        self.is_fitted = True


