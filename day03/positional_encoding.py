"""
Day 03 — Sinusoidal Positional Encoding
=========================================
Implements the original "Attention Is All You Need" positional encoding.
Generates encodings for positions 0-49, dimensions 0-63, plots a heatmap,
and prints sample encodings step by step.

Formula:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Even dimensions → sine,  Odd dimensions → cosine.
Each position gets a unique "fingerprint" the model can learn to read.

Usage:
    python3 positional_encoding.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

def sinusoidal_pe(max_positions: int, d_model: int) -> np.ndarray:
    PE       = np.zeros((max_positions, d_model))
    positions = np.arange(max_positions)[:, np.newaxis]
    dims      = np.arange(0, d_model, 2)
    div_term  = np.power(10000.0, dims / d_model)
    PE[:, 0::2] = np.sin(positions / div_term)
    PE[:, 1::2] = np.cos(positions / div_term)
    return PE


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MAX_POS = 50
D_MODEL = 64

print("=" * 62)
print("  POSITIONAL ENCODING  —  how the model knows word order")
print("=" * 62)
print("""
  Problem: transformers process all tokens at once (in parallel).
  Unlike RNNs, they have no built-in sense of sequence order.
  Solution: add a unique "position fingerprint" to each token's
  embedding before it enters the transformer.

  The fingerprint is a vector of sine and cosine waves at
  different frequencies — one frequency per pair of dimensions.
""")

# ---------------------------------------------------------------------------
# Step 1: explain the formula dimension by dimension
# ---------------------------------------------------------------------------

print("=" * 62)
print("  STEP 1 — THE FORMULA, dimension by dimension")
print("=" * 62)
print(f"""
  For each position (0 to {MAX_POS-1}) and each dimension (0 to {D_MODEL-1}):

    Even dim 2i   → sin(pos / 10000^(2i/{D_MODEL}))
    Odd  dim 2i+1 → cos(pos / 10000^(2i/{D_MODEL}))

  The divisor 10000^(2i/d_model) grows as i increases.
  Small i → small divisor → fast oscillation (changes quickly with pos)
  Large i → large divisor → slow oscillation (changes slowly with pos)

  Showing divisor values for selected dimension pairs:
""")

print(f"  {'Dim pair (2i)':<16} {'Divisor':>12}  {'Oscillation speed'}")
print(f"  {'─'*14:<16} {'─'*10:>12}  {'─'*20}")
for i in [0, 1, 2, 4, 8, 16, 31]:
    dim   = 2 * i
    div   = 10000.0 ** (dim / D_MODEL)
    speed = "fast  ████████" if div < 10 else \
            "medium ████    " if div < 1000 else \
            "slow   █       "
    print(f"  dim {dim:<2} and {dim+1:<2} ({i:>2})   {div:>12.2f}  {speed}")

# ---------------------------------------------------------------------------
# Step 2: compute and show raw values building up
# ---------------------------------------------------------------------------

PE = sinusoidal_pe(MAX_POS, D_MODEL)

print(f"\n\n{'='*62}")
print("  STEP 2 — BUILDING THE ENCODING for position 1")
print("=" * 62)
print("""
  Let's watch position 1's fingerprint get built up,
  dimension by dimension (showing first 16 dims):
""")

pos = 1
print(f"  {'Dim':<6} {'Type':<6} {'Formula':<32} {'Value':>8}")
print(f"  {'─'*4:<6} {'─'*4:<6} {'─'*30:<32} {'─'*6:>8}")
for dim in range(16):
    i      = dim // 2
    div    = 10000.0 ** ((2 * i) / D_MODEL)
    if dim % 2 == 0:
        formula = f"sin({pos} / {div:.2f})"
        val     = np.sin(pos / div)
        kind    = "sin"
    else:
        formula = f"cos({pos} / {div:.2f})"
        val     = np.cos(pos / div)
        kind    = "cos"
    bar = "▓" * int(abs(val) * 10)
    sign = "+" if val >= 0 else "-"
    print(f"  {dim:<6} {kind:<6} {formula:<32} {val:>+8.4f}  {bar}")

# ---------------------------------------------------------------------------
# Step 3: compare positions side by side
# ---------------------------------------------------------------------------

print(f"\n\n{'='*62}")
print("  STEP 3 — COMPARING POSITIONS  (first 12 dimensions)")
print("=" * 62)
print("""
  Each position gets a unique pattern. Nearby positions look
  similar in the slow dimensions but differ in the fast ones —
  that's how the model distinguishes them.
""")

compare_positions = [0, 1, 2, 5, 10, 25, 49]
print(f"  {'Pos':<6}", end="")
for d in range(12):
    print(f"  dim{d:<2}", end="")
print()
print(f"  {'─'*4:<6}", end="")
for d in range(12):
    print(f"  {'─'*4}", end="")
print()

for pos in compare_positions:
    print(f"  {pos:<6}", end="")
    for d in range(12):
        val = PE[pos, d]
        # Show as a simple +/- bar for readability
        block = "██" if val > 0.5 else "▒▒" if val > 0 else "░░" if val > -0.5 else "  "
        print(f"  {val:>+.2f}", end="")
    print()

print("""
  Reading the table: each row is a position's fingerprint.
  Rows 0 and 1 look very similar in slow dims (right columns)
  but differ in fast dims (left columns) — unique at fine grain.
  Rows 0 and 49 differ everywhere — very distinct fingerprints.
""")

# ---------------------------------------------------------------------------
# Step 4: full position encodings for 0, 1, 10, 49 (all 64 dims)
# ---------------------------------------------------------------------------

print("=" * 62)
print("  STEP 4 — FULL 64-DIM ENCODING for key positions")
print("=" * 62)

for pos in [0, 1, 10, 49]:
    values = PE[pos]
    print(f"\n  Position {pos}:")
    # Print 8 values per row
    for row_start in range(0, D_MODEL, 8):
        chunk = values[row_start:row_start+8]
        formatted = "  ".join(f"{v:+.3f}" for v in chunk)
        print(f"    dims {row_start:>2}–{row_start+7:<2}: [{formatted}]")

# ---------------------------------------------------------------------------
# Step 5: heatmap plot
# ---------------------------------------------------------------------------

print(f"\n\n{'='*62}")
print("  STEP 5 — SAVING pe_heatmap.png")
print("=" * 62)
print("""
  Left panel  : heatmap of the full 50×64 PE matrix
                Red = positive, Blue = negative
                Notice vertical stripes (fast dims, left) vs
                gradual shifts (slow dims, right)

  Right panel : 4 specific dimensions plotted as sine/cosine waves
                across all 50 positions — shows the oscillation speeds
""")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("#0f1117")

ax = axes[0]
ax.set_facecolor("#0f1117")
im = ax.imshow(PE, aspect="auto", cmap="RdBu", vmin=-1, vmax=1,
               interpolation="nearest")
plt.colorbar(im, ax=ax, label="Encoding value", shrink=0.8)
ax.set_title("Sinusoidal Positional Encoding\n(positions 0–49, dims 0–63)",
             color="#f8fafc", fontsize=11)
ax.set_xlabel("Dimension", color="#94a3b8")
ax.set_ylabel("Position", color="#94a3b8")
ax.tick_params(colors="#64748b")
for spine in ax.spines.values():
    spine.set_edgecolor("#2d3348")

ax2 = axes[1]
ax2.set_facecolor("#0f1117")
colors = ["#6366f1", "#f472b6", "#4ade80", "#fbbf24"]
for dim, color in zip([0, 1, 4, 10], colors):
    label = f"dim {dim} ({'sin' if dim % 2 == 0 else 'cos'})"
    ax2.plot(PE[:, dim], color=color, linewidth=1.8, label=label)

ax2.set_title("PE values across positions\n(selected dimensions)",
              color="#f8fafc", fontsize=11)
ax2.set_xlabel("Position", color="#94a3b8")
ax2.set_ylabel("Value", color="#94a3b8")
ax2.legend(facecolor="#1e2130", edgecolor="#2d3348",
           labelcolor="#e2e8f0", fontsize=9)
ax2.tick_params(colors="#64748b")
ax2.axhline(0, color="#2d3348", linewidth=0.8, linestyle="--")
for spine in ax2.spines.values():
    spine.set_edgecolor("#2d3348")

plt.suptitle("Transformer Sinusoidal Positional Encoding — Day 03",
             color="#f8fafc", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("pe_heatmap.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()

print("  Saved: pe_heatmap.png\n")
print("=" * 62)
print("  KEY TAKEAWAYS")
print("=" * 62)
print("""
  1. Every position gets a unique 64-number fingerprint
  2. Position 0 starts with all cosines (sin(0)=0 everywhere)
  3. Nearby positions differ mostly in fast (low) dimensions
  4. Far apart positions differ in all dimensions
  5. The model learns to read these patterns during training —
     it figures out that "dim 0 oscillates fast = fine position,
     dim 60 oscillates slow = coarse position"
""")
