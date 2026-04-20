"""
Day 03 — Embedding Space Visualizer
=====================================
Loads embeddings.npy trained by embeddings.py, projects to 2D with PCA,
and plots a labelled scatter plot saved as embedding_space.png.
Also prints top-5 nearest neighbors for selected words.

Usage:
    python3 visualize.py
    (run embeddings.py first to generate embeddings.npy and vocab.txt)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print("=" * 62)
print("  EMBEDDING VISUALIZER  —  collapsing 32D vectors to 2D")
print("=" * 62)

embeddings = np.load("embeddings.npy")
with open("vocab.txt") as f:
    words = [line.strip() for line in f]

assert len(words) == embeddings.shape[0], \
    f"Vocab size mismatch: {len(words)} words vs {embeddings.shape[0]} rows"

print(f"""
  Loaded: {embeddings.shape[0]} words × {embeddings.shape[1]} dimensions
  Each word is a point in {embeddings.shape[1]}-dimensional space.
  We can't visualize {embeddings.shape[1]}D, so PCA compresses it to 2D
  while preserving as much structure as possible.
""")

# ---------------------------------------------------------------------------
# Cosine similarity helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def nearest_neighbors(word: str, top_n: int = 5) -> list:
    if word not in words:
        return []
    idx = words.index(word)
    vec = embeddings[idx]
    sims = [(words[j], cosine_similarity(vec, embeddings[j]))
            for j in range(len(words)) if j != idx]
    sims.sort(key=lambda x: -x[1])
    return sims[:top_n]


# ---------------------------------------------------------------------------
# Nearest neighbors — step by step
# ---------------------------------------------------------------------------

probe_words = ["model", "token", "training", "attention", "embedding"]

print("=" * 62)
print("  STEP 1 — NEAREST NEIGHBORS in full 32D space")
print("=" * 62)
print("""
  Before we flatten to 2D, we compute real cosine similarity
  in the full 32D space. This is the ground truth — the 2D
  plot is an approximation of this.
""")

for word in probe_words:
    neighbors = nearest_neighbors(word)
    if not neighbors:
        print(f"  '{word}' not in vocabulary\n")
        continue
    print(f"  ┌─ '{word}' — most similar words in 32D space:")
    for rank, (neighbor, score) in enumerate(neighbors, 1):
        bar = "▓" * int(score * 20) if score > 0 else ""
        print(f"  │  #{rank}  {neighbor:16} cosine={score:.3f}  {bar}")
    print()

# ---------------------------------------------------------------------------
# PCA step by step
# ---------------------------------------------------------------------------

print("=" * 62)
print("  STEP 2 — PCA PROJECTION from 32D → 2D")
print("=" * 62)
print("""
  PCA finds the 2 directions in 32D space that capture the
  most variance — the axes along which words spread out most.

  PC1 = direction of greatest spread
  PC2 = direction of second greatest spread (perpendicular to PC1)

  Words close together in 32D will (mostly) stay close in 2D.
  Some distance is lost — that's the information PCA discards.
""")

pca    = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(embeddings)

pc1_var = pca.explained_variance_ratio_[0]
pc2_var = pca.explained_variance_ratio_[1]
total   = pc1_var + pc2_var

print(f"  PC1 captures : {pc1_var:.1%} of total variance")
print(f"  PC2 captures : {pc2_var:.1%} of total variance")
print(f"  Together     : {total:.1%} — the rest is lost in compression")
print(f"\n  Showing where probe words landed in 2D:")
print(f"\n  {'Word':<14} {'2D x':>8}  {'2D y':>8}")
print(f"  {'─'*12:<14} {'─'*6:>8}  {'─'*6:>8}")
for word in probe_words:
    if word not in words:
        continue
    i = words.index(word)
    print(f"  {word:<14} {coords[i,0]:>8.3f}  {coords[i,1]:>8.3f}")

# ---------------------------------------------------------------------------
# Show all words with their 2D coordinates
# ---------------------------------------------------------------------------

print(f"\n  All {len(words)} words in 2D (sorted by x position):")
print(f"\n  {'Word':<16} {'x':>8}  {'y':>8}")
print(f"  {'─'*14:<16} {'─'*6:>8}  {'─'*6:>8}")
sorted_by_x = sorted(zip(words, coords), key=lambda t: t[1][0])
for word, (x, y) in sorted_by_x:
    marker = "  ◀ probe" if word in probe_words else ""
    print(f"  {word:<16} {x:>8.3f}  {y:>8.3f}{marker}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

print("\n" + "=" * 62)
print("  STEP 3 — PLOTTING embedding_space.png")
print("=" * 62)
print("\n  Rendering scatter plot with all words labelled...")

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

ax.scatter(coords[:, 0], coords[:, 1],
           s=30, color="#6366f1", alpha=0.7, zorder=3)

for i, word in enumerate(words):
    ax.annotate(
        word,
        xy=(coords[i, 0], coords[i, 1]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
        color="#e2e8f0",
        alpha=0.9,
    )

for word in probe_words:
    if word not in words:
        continue
    i = words.index(word)
    ax.scatter(coords[i, 0], coords[i, 1], s=120, color="#f472b6", zorder=5)
    ax.annotate(
        word,
        xy=(coords[i, 0], coords[i, 1]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="#f472b6",
    )

ax.set_title("Word Embedding Space — PCA Projection (Day 03)",
             color="#f8fafc", fontsize=13, pad=12)
ax.set_xlabel(f"PC1 ({pc1_var:.1%} variance)", color="#94a3b8")
ax.set_ylabel(f"PC2 ({pc2_var:.1%} variance)", color="#94a3b8")
ax.tick_params(colors="#64748b")
for spine in ax.spines.values():
    spine.set_edgecolor("#2d3348")

plt.tight_layout()
plt.savefig("embedding_space.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()

print("""
  Saved: embedding_space.png

  How to read the plot:
  → Words close together  = similar meaning / similar context
  → Words far apart       = used in different parts of the corpus
  → Pink dots             = probe words (model, token, training...)
  → Purple dots           = all other words

  Remember: the axes (PC1, PC2) have no inherent meaning —
  only the distances between words matter.
""")
