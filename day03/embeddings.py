"""
Day 03 — Skip-gram Word Embedding Trainer
==========================================
Trains word2vec-style embeddings using pure PyTorch (no gensim).
Uses MPS backend on M1 Mac for GPU acceleration.

Skip-gram: for each center word, predict surrounding context words.
The model learns dense vectors where similar-context words cluster together.

Usage:
    python3 embeddings.py
"""

import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

# M1 Mac: use MPS if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Same corpus as BPE tokenizer
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
# Preprocessing
# ---------------------------------------------------------------------------

def build_vocab(corpus: str, min_count: int = 1):
    """Build word→index mapping from corpus."""
    words = re.findall(r"[a-z]+", corpus.lower())
    counts = Counter(words)
    vocab = [w for w, c in counts.items() if c >= min_count]
    vocab = sorted(set(vocab))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word, words


def build_skipgram_pairs(words: list, word2idx: dict,
                          window: int = 2) -> list:
    """
    Generate (center_idx, context_idx) training pairs.
    For each word, pair it with every word within ±window positions.
    """
    pairs = []
    indexed = [word2idx[w] for w in words if w in word2idx]
    for i, center in enumerate(indexed):
        lo = max(0, i - window)
        hi = min(len(indexed), i + window + 1)
        for j in range(lo, hi):
            if j != i:
                pairs.append((center, indexed[j]))
    return pairs


def show_skipgram_pairs(words: list, word2idx: dict,
                         idx2word: dict, window: int = 2,
                         sample_sentence: str = "the model learns from data") -> None:
    """
    Visually walk through skip-gram pair generation for one sentence,
    showing exactly which (center, context) pairs get created.
    """
    print("=" * 62)
    print("  SKIP-GRAM PAIR GENERATION  —  what the model trains on")
    print("=" * 62)
    print(f"\n  Window size = {window}  (look ±{window} words in each direction)")
    print(f"\n  Example sentence: {sample_sentence!r}\n")

    sample_words = [w for w in sample_sentence.split() if w in word2idx]
    print(f"  {'Position':<10} {'Center word':<14} Context words within window")
    print(f"  {'─'*8:<10} {'─'*12:<14} {'─'*30}")

    for i, center in enumerate(sample_words):
        lo = max(0, i - window)
        hi = min(len(sample_words), i + window + 1)
        context = [sample_words[j] for j in range(lo, hi) if j != i]
        pairs_str = "  ".join(f"({center!r}, {c!r})" for c in context)
        print(f"  {i:<10} {center:<14} → {pairs_str}")


# ---------------------------------------------------------------------------
# Skip-gram Model
# ---------------------------------------------------------------------------

class SkipGram(nn.Module):
    """
    Two embedding tables: one for center words, one for context words.
    Dot product of center and context vectors → binary classification
    (real pair vs noise pair) via negative sampling loss.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.center_emb  = nn.Embedding(vocab_size, embed_dim)
        self.context_emb = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.center_emb.weight,  -0.1, 0.1)
        nn.init.uniform_(self.context_emb.weight, -0.1, 0.1)

    def forward(self, center: torch.Tensor,
                context: torch.Tensor,
                negatives: torch.Tensor) -> torch.Tensor:
        v_center  = self.center_emb(center)
        u_pos     = self.context_emb(context)
        u_neg     = self.context_emb(negatives)
        pos_score = torch.sum(v_center * u_pos, dim=1)
        pos_loss  = torch.nn.functional.logsigmoid(pos_score)
        neg_score = torch.bmm(u_neg, v_center.unsqueeze(2)).squeeze(2)
        neg_loss  = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)
        return -(pos_loss + neg_loss).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(corpus: str,
          embed_dim: int = 32,
          window: int = 2,
          neg_samples: int = 5,
          epochs: int = 50,
          batch_size: int = 64,
          lr: float = 0.01) -> tuple:
    """Train skip-gram embeddings. Returns (model, word2idx, idx2word)."""

    word2idx, idx2word, words = build_vocab(corpus)
    vocab_size = len(word2idx)
    pairs = build_skipgram_pairs(words, word2idx, window=window)

    # Show pair generation before training
    show_skipgram_pairs(words, word2idx, idx2word, window=window)

    print("\n" + "=" * 62)
    print("  TRAINING  —  adjusting vectors to predict context words")
    print("=" * 62)
    print(f"\n  Device        : {device}")
    print(f"  Vocab size    : {vocab_size} unique words")
    print(f"  Training pairs: {len(pairs)}  (center→context combinations)")
    print(f"  Embed dim     : {embed_dim}D  (each word = {embed_dim} numbers)")
    print(f"  Neg samples   : {neg_samples}  (random wrong pairs per real pair)")
    print(f"  Epochs        : {epochs}")
    print(f"\n  What's happening each epoch:")
    print(f"  → shuffle all pairs")
    print(f"  → for each (center, context) pair:")
    print(f"       • pull center vector and context vector")
    print(f"       • compute dot product — high = predicted neighbors")
    print(f"       • compare to {neg_samples} random wrong pairs")
    print(f"       • nudge vectors so real pairs score higher")
    print(f"  → loss should decrease as vectors improve\n")

    model     = SkipGram(vocab_size, embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    centers  = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    contexts = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    for epoch in range(epochs):
        perm     = torch.randperm(len(pairs))
        centers  = centers[perm]
        contexts = contexts[perm]

        total_loss = 0.0
        steps = 0

        for i in range(0, len(pairs), batch_size):
            c_batch = centers[i:i+batch_size].to(device)
            x_batch = contexts[i:i+batch_size].to(device)
            negs    = torch.randint(0, vocab_size,
                                    (len(c_batch), neg_samples), device=device)
            optimizer.zero_grad()
            loss = model(c_batch, x_batch, negs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        if (epoch + 1) % 10 == 0:
            avg = total_loss / steps
            bar = "█" * int((1 - min(avg / 3.0, 1.0)) * 20)
            print(f"  Epoch {epoch+1:>3}/{epochs} — loss: {avg:.4f}  {bar}")

    print("\n  Training complete — vectors now encode word context similarity.")
    return model, word2idx, idx2word


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("  WORD EMBEDDING TRAINER  —  turning words into vectors")
    print("=" * 62)
    print("""
  Idea: words that appear in similar contexts get similar vectors.
  "model" and "network" both appear near "training", "loss", "layer"
  → after training their vectors will point in similar directions.

  We use skip-gram: given a center word, predict its neighbors.
  The vectors are just random numbers at first — training nudges
  them until the dot product of real (center, context) pairs is
  higher than random (center, noise) pairs.
""")

    model, word2idx, idx2word = train(CORPUS)

    # Extract final embeddings
    embeddings = model.center_emb.weight.detach().cpu().numpy()

    # Save for visualize.py
    np.save("embeddings.npy", embeddings)
    with open("vocab.txt", "w") as f:
        for i in range(len(idx2word)):
            f.write(idx2word[i] + "\n")

    print(f"\n  Saved embeddings.npy — shape: {embeddings.shape}")
    print(f"  Saved vocab.txt       — {len(idx2word)} words")
    print(f"\n  Each word is now a row of {embeddings.shape[1]} numbers.")
    print(f"  Words used in similar sentences → similar rows (vectors).\n")

    # Show raw vectors for a few words
    print("=" * 62)
    print("  RAW VECTORS  —  what the numbers look like (first 8 dims)")
    print("=" * 62)
    sample_words = ["model", "token", "training", "attention"]
    for word in sample_words:
        if word not in word2idx:
            continue
        vec = embeddings[word2idx[word]][:8]
        formatted = "  ".join(f"{v:+.3f}" for v in vec)
        print(f"\n  {word:12} [{formatted} ...]")
    print("\n  These numbers mean nothing on their own — only the")
    print("  distances between vectors carry meaning.\n")

    # Nearest neighbors with step-by-step explanation
    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    probe_words = ["model", "token", "training", "attention", "embedding"]
    print("=" * 62)
    print("  NEAREST NEIGHBORS  —  cosine similarity between vectors")
    print("=" * 62)
    print("\n  Cosine similarity = angle between two vectors")
    print("  1.0 = identical direction,  0.0 = unrelated,  -1.0 = opposite\n")

    for word in probe_words:
        if word not in word2idx:
            continue
        idx = word2idx[word]
        vec = embeddings[idx]
        sims = [(idx2word[j], cosine_sim(vec, embeddings[j]))
                for j in range(len(idx2word)) if j != idx]
        sims.sort(key=lambda x: -x[1])
        top5 = sims[:5]
        print(f"  ┌─ '{word}'")
        for neighbor, score in top5:
            bar = "▓" * int(score * 20) if score > 0 else ""
            print(f"  │  {neighbor:16} {score:+.3f}  {bar}")
        print(f"  └─ (all {len(sims)} other words compared)\n")


if __name__ == "__main__":
    main()
