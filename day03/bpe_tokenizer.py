"""
Day 03 — BPE Tokenizer from Scratch
=====================================
Byte Pair Encoding: the algorithm behind GPT-2/4 and Claude tokenizers.
No tokenizer libraries — pure Python stdlib only.

Algorithm:
  1. Split corpus into characters (+ </w> end-of-word marker)
  2. Count every adjacent symbol pair
  3. Merge the most frequent pair into a new symbol
  4. Repeat until vocab reaches target size

Usage:
    python3 bpe_tokenizer.py
"""

import re
from collections import Counter

# ---------------------------------------------------------------------------
# Corpus — 20 sentences covering NLP/ML vocabulary
# ---------------------------------------------------------------------------
CORPUS = """
the cat sat on the mat
the cat ate the rat
the rat ran fast
a fast cat ran past
low lower newest wider
the model learns from data
language models predict tokens
transformers use attention layers
embeddings encode word meaning
the tokenizer splits text into tokens
byte pair encoding merges frequent pairs
the vocabulary grows with each merge step
training loss decreases over epochs
gradient descent updates model weights
the context window limits token count
neural networks learn representations
the embedding layer maps tokens to vectors
attention scores weigh token importance
residual connections improve gradient flow
layer normalization stabilizes training
"""

# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    def __init__(self):
        self.merges = []          # list of (pair_tuple → merged_string)
        self.vocab = {}           # token_string → integer ID
        self.id_to_token = {}     # integer ID → token_string

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, corpus: str, vocab_size: int,
              trace_words: list = None) -> None:
        """
        Run BPE merges on corpus until vocab reaches vocab_size.
        Populates self.merges, self.vocab, self.id_to_token.

        trace_words: optional list of words to visually track through
                     every merge step (e.g. ["token", "lower"])
        """
        print("=" * 62)
        print("  BPE TRAINING  —  watching raw text become a vocabulary")
        print("=" * 62)
        print(f"\nTarget vocab size : {vocab_size}")
        print(f"Corpus size       : {len(corpus.split())} words\n")

        # Step 1: build word frequency table
        # Each word is stored as a tuple of chars + </w>
        # "token" → ('t','o','k','e','n','</w>')
        word_freqs = self._build_word_freqs(corpus)

        # Step 2: collect initial character vocab
        base_vocab = set()
        for word in word_freqs:
            for ch in word:
                base_vocab.add(ch)

        print("─" * 62)
        print("STEP 0 — Starting point: every character is its own token")
        print("─" * 62)
        print(f"  Characters in corpus : {sorted(base_vocab)}")
        print(f"  Vocab size right now : {len(base_vocab)}")

        # Show how a few words look before any merging
        if trace_words:
            print(f"\n  How trace words look at the start (just characters):")
            for tw in trace_words:
                key = tuple(list(tw) + ["</w>"])
                if key in word_freqs:
                    print(f"    '{tw}' → {list(key)}")
        print()

        # Step 3: merge until vocab_size is reached
        num_merges = vocab_size - len(base_vocab)
        for step in range(num_merges):
            pair_counts = self._count_pairs(word_freqs)
            if not pair_counts:
                print("No more pairs to merge.")
                break

            # Show top-3 candidate pairs before picking the winner
            top3 = sorted(pair_counts.items(), key=lambda x: -x[1])[:3]
            top3_str = "  |  ".join(
                f"'{a}'+'{b}' (×{c})" for (a, b), c in top3
            )

            best   = top3[0][0]
            merged = best[0] + best[1]
            self.merges.append((best, merged))

            print(f"── Merge step {step+1:>3} ─────────────────────────────────────")
            print(f"  Top candidates : {top3_str}")
            print(f"  Winner         : '{best[0]}' + '{best[1]}' → '{merged}'")

            word_freqs = self._apply_merge(word_freqs, best, merged)

            # Show which words in the corpus were affected by this merge
            affected = [
                " ".join(w) for w in word_freqs
                if merged in w
            ]
            if affected:
                sample = affected[:4]
                more   = f"  (+{len(affected)-4} more)" if len(affected) > 4 else ""
                print(f"  Affected words : {sample}{more}")

            # Track how our trace words look after this merge
            if trace_words:
                for tw in trace_words:
                    key = None
                    # find current representation of this word in word_freqs
                    for w in word_freqs:
                        raw = "".join(w).replace("</w>", "")
                        if raw == tw:
                            key = w
                            break
                    if key:
                        current = list(key)
                        if merged in current:   # only print if this merge touched it
                            print(f"  '{tw}' is now  : {current}")
            print()

        # Step 4: build vocab from all symbols present after merging
        all_tokens = set()
        all_tokens.add("<unk>")
        for word in word_freqs:
            for sym in word:
                all_tokens.add(sym)

        self.vocab       = {tok: i for i, tok in enumerate(sorted(all_tokens))}
        self.id_to_token = {i: tok for tok, i in self.vocab.items()}

        print("=" * 62)
        print(f"  TRAINING DONE  —  final vocab size: {len(self.vocab)}")
        print("=" * 62)

    def _build_word_freqs(self, corpus: str) -> Counter:
        """
        Split corpus into words, represent each word as a tuple of
        characters with a </w> end-of-word marker.
        e.g. "low" → ('l','o','w','</w>')
        """
        words = re.findall(r"[a-z]+", corpus.lower())
        freq: Counter = Counter()
        for w in words:
            key = tuple(list(w) + ["</w>"])
            freq[key] += 1
        return freq

    def _count_pairs(self, word_freqs: Counter) -> Counter:
        """Count all adjacent symbol pairs weighted by word frequency."""
        counts: Counter = Counter()
        for word, freq in word_freqs.items():
            for a, b in zip(word, word[1:]):
                counts[(a, b)] += freq
        return counts

    def _apply_merge(self, word_freqs: Counter,
                     pair: tuple, merged: str) -> Counter:
        """Replace all occurrences of pair in every word with merged symbol."""
        new_freqs: Counter = Counter()
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_freqs[tuple(new_word)] += freq
        return new_freqs

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, text: str, verbose: bool = False) -> list[int]:
        """
        Tokenize text using learned merge rules → list of integer token IDs.
        Unknown symbols map to <unk>.

        verbose=True prints every merge step for every word so you can
        watch the characters collapse into tokens in real time.
        """
        token_ids = []
        all_words = re.findall(r"[a-z]+", text.lower())

        if verbose:
            print(f"\n  Encoding: {text!r}")
            print(f"  Words to process: {all_words}\n")

        for word in all_words:
            symbols = list(word) + ["</w>"]

            if verbose:
                print(f"  ┌─ Word: '{word}'")
                print(f"  │  Start : {symbols}")

            # Apply every merge rule in the order it was learned
            prev = None
            for (a, b), merged in self.merges:
                i = 0
                new_syms = []
                while i < len(symbols):
                    if i < len(symbols)-1 and symbols[i] == a and symbols[i+1] == b:
                        new_syms.append(merged)
                        i += 2
                    else:
                        new_syms.append(symbols[i])
                        i += 1
                # Only print when something actually changed
                if verbose and new_syms != symbols:
                    print(f"  │  + merge '{a}'+'{b}' → '{merged}' : {new_syms}")
                symbols = new_syms

            ids = [self.vocab.get(sym, self.vocab["<unk>"]) for sym in symbols]
            token_ids.extend(ids)

            if verbose:
                print(f"  │  Final tokens : {symbols}")
                print(f"  └─ Token IDs    : {ids}\n")

        return token_ids

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, token_ids: list[int]) -> str:
        """
        Convert list of integer token IDs back to a string.
        Strips </w> markers and reconstructs spaces between words.
        """
        tokens = [self.id_to_token.get(i, "<unk>") for i in token_ids]
        text = "".join(tokens)
        # </w> marks word boundaries — replace with space
        text = text.replace("</w>", " ").strip()
        return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tok = BPETokenizer()

    # trace_words: watch these words get built up merge-by-merge during training
    tok.train(CORPUS, vocab_size=80, trace_words=["token", "lower", "model"])

    # ── Verbose encoding: watch every word collapse into tokens ──────────
    print("\n" + "=" * 62)
    print("  ENCODING WALKTHROUGH  —  characters merging into tokens")
    print("=" * 62)

    test_sentences = [
        "the cat sat on the mat",
        "low lower newest",
        "the model learns embeddings",
        "transformers use attention",
    ]

    for sentence in test_sentences:
        print(f"\n{'─' * 62}")
        print(f"  Sentence: {sentence!r}")
        print(f"{'─' * 62}")
        ids     = tok.encode(sentence, verbose=True)
        decoded = tok.decode(ids)
        tokens  = [tok.id_to_token[i] for i in ids]
        print(f"  → Final tokens : {tokens}")
        print(f"  → IDs          : {ids}")
        print(f"  → Decoded      : {decoded!r}")

    print("\n" + "=" * 62)
    print(f"  FULL VOCABULARY ({len(tok.vocab)} tokens)")
    print("=" * 62)
    for tok_str, tok_id in tok.vocab.items():
        print(f"  {tok_id:>4}: {tok_str!r}")


if __name__ == "__main__":
    main()
