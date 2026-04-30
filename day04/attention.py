"""
Day 04 — Attention Mechanisms
==============================
Builds attention from the ground up in two stages:

  Stage 1 — SingleHeadAttention
    One set of Q/K/V projections, one attention pattern.
    Easiest to read and reason about.

  Stage 2 — MultiHeadAttention
    n_heads independent sub-spaces running in parallel, then concat + project.
    Each head can learn a different relationship (syntax, coreference, position…).

Both use CAUSAL MASKING — position i can only attend to positions ≤ i.
That's what makes this "GPT-style" (decoder-only), not BERT-style (bidirectional).

Concept recap from Claude.ai:
  Q (Query)  — "what am I looking for?"
  K (Key)    — "what do I advertise about myself?"
  V (Value)  — "what do I actually contribute if selected?"
  scores     — QKᵀ / √d_k : how much each position 'wants' each other position
  causal mask — zero out future positions (set to -inf before softmax)
  weights    — softmax(scores) : sum to 1 per row, used to mix V rows
  output     — weights @ V : weighted blend of all value vectors

Usage:
    python3 attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Stage 1 — Single-Head Attention
# ---------------------------------------------------------------------------

class SingleHeadAttention(nn.Module):
    """
    The simplest possible attention: one Q, K, V projection each.

    d_model → projection → d_head  (often d_head == d_model for single head)

    Steps inside forward():
      1. Project x into Q, K, V spaces
      2. Compute raw scores = Q @ Kᵀ, scaled by 1/√d_head
      3. Apply causal mask (upper triangle → -inf)
      4. Softmax → attention weights (each row sums to 1)
      5. Weighted sum: output = weights @ V
    """

    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.d_head = d_head
        self.scale  = d_head ** -0.5  # 1/√d_head — prevents dot products from growing huge

        # No bias: standard practice in GPT-style transformers
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x : (B, T, d_model)  — batch, sequence length, model dim
        Returns:
          out     : (B, T, d_head)  — context-enriched vectors
          weights : (B, T, T)       — attention pattern (useful for visualization)
        """
        B, T, _ = x.shape

        Q = self.W_q(x)  # (B, T, d_head) — queries
        K = self.W_k(x)  # (B, T, d_head) — keys
        V = self.W_v(x)  # (B, T, d_head) — values

        # ── Score: how much does each query "want" each key? ──────────────
        # QKᵀ gives a T×T matrix: scores[i,j] = how much position i attends to j
        scores = Q @ K.transpose(-2, -1) * self.scale  # (B, T, T)

        # ── Causal mask: position i must not see positions j > i ──────────
        # torch.triu(..., diagonal=1) gives upper-triangle ones (future positions)
        # We fill those with -inf so softmax → 0 for those positions
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

        # ── Softmax: convert scores to a probability distribution ─────────
        # Each row now sums to 1; the -inf positions become 0
        weights = F.softmax(scores, dim=-1)  # (B, T, T)

        # ── Weighted sum of values ─────────────────────────────────────────
        # For each position, mix the V vectors using the attention weights
        out = weights @ V  # (B, T, d_head)

        return out, weights


# ---------------------------------------------------------------------------
# Stage 2 — Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    n_heads attention heads running in parallel, each in a d_head-dim subspace.
    d_head = d_model // n_heads (they partition the model dimension evenly).

    Why multiple heads?
      Each head learns a different attention pattern.
      Head 1 might track "what noun does this verb refer to?"
      Head 2 might track "is the previous token a preposition?"
      They run simultaneously, then their outputs are concatenated and projected.

    Implementation trick:
      Instead of n_heads separate Linear layers, we use one big W_q/W_k/W_v
      and reshape the output into (B, H, T, d_head) — same math, more efficient.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head  = d_model // n_heads  # each head works in this many dims
        self.scale   = self.d_head ** -0.5

        # Single big projection per Q/K/V — we'll split into heads via reshape
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection: recombine all heads back into d_model
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x : (B, T, d_model)
        Returns:
          out     : (B, T, d_model)  — same shape as input (ready for residual)
          weights : (B, H, T, T)     — one attention map per head
        """
        B, T, C = x.shape  # C == d_model

        # ── Project + split into heads ────────────────────────────────────
        # After W_q: (B, T, d_model)
        # After view: (B, T, n_heads, d_head)
        # After transpose(1,2): (B, n_heads, T, d_head)  ← head dim second for bmm
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # ── Scaled dot-product attention (all heads in parallel) ──────────
        scores = Q @ K.transpose(-2, -1) * self.scale  # (B, H, T, T)

        # Causal mask — same mask for every head and every batch item
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        weights = F.softmax(scores, dim=-1)  # (B, H, T, T)
        out     = weights @ V                # (B, H, T, d_head)

        # ── Reassemble heads ──────────────────────────────────────────────
        # transpose back to (B, T, H, d_head), then merge H and d_head → d_model
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, d_model)

        # Final linear mixes information across heads
        out = self.W_o(out)

        return out, weights


# ---------------------------------------------------------------------------
# Standalone demo & tests
# ---------------------------------------------------------------------------

def print_section(title: str):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def demo_single_head():
    print_section("STAGE 1 — Single-Head Attention")

    d_model = 32
    d_head  = 32   # single head: head dim == model dim
    B, T    = 1, 6  # one sentence, 6 tokens

    x    = torch.randn(B, T, d_model)
    attn = SingleHeadAttention(d_model, d_head)

    with torch.no_grad():
        out, weights = attn(x)

    print(f"\n  Input  shape : {list(x.shape)}      (batch=1, seq=6, dim=32)")
    print(f"  Output shape : {list(out.shape)}     (same seq and dim — ready for residual)")
    print(f"  Weights shape: {list(weights.shape)} (6×6 attention pattern)")

    print("\n  Attention weight matrix (row i = how much position i attends to each j):")
    print("  Causal mask → upper triangle should be 0.0\n")
    fmt = weights[0].numpy()
    header = "       " + "".join(f"  pos{j}" for j in range(T))
    print(f"  {header}")
    for i in range(T):
        row = "  ".join(f"{v:.3f}" for v in fmt[i])
        print(f"  pos{i}  {row}")

    print("\n  ✓ Each row sums to 1.0 (it's a probability distribution)")
    print("  ✓ Upper-right triangle is 0.0 (future positions masked out)")
    row_sums = weights[0].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(T), atol=1e-5), "Row sums != 1"
    assert (weights[0].triu(diagonal=1).abs() < 1e-5).all(), "Causal mask violated"
    print("  ✓ Assertions passed")


def demo_multi_head():
    print_section("STAGE 2 — Multi-Head Attention (8 heads)")

    d_model = 32
    n_heads = 8        # 32 / 8 = 4 dims per head
    B, T    = 2, 10    # two sentences, 10 tokens each

    x    = torch.randn(B, T, d_model)
    attn = MultiHeadAttention(d_model, n_heads)

    with torch.no_grad():
        out, weights = attn(x)

    print(f"\n  d_model={d_model}, n_heads={n_heads} → d_head={d_model//n_heads} dims per head")
    print(f"\n  Input  shape  : {list(x.shape)}")
    print(f"  Output shape  : {list(out.shape)}     (same as input — drop-in residual)")
    print(f"  Weights shape : {list(weights.shape)}  (batch, heads, seq, seq)")

    print("\n  Head-by-head row-sum check (each row must sum to 1.0):")
    for h in range(n_heads):
        row_sums = weights[0, h].sum(dim=-1)
        ok = torch.allclose(row_sums, torch.ones(T), atol=1e-5)
        print(f"    Head {h}: all row sums == 1.0 → {'✓' if ok else '✗'}")

    print("\n  Causal constraint check (no future peeking) per head:")
    for h in range(n_heads):
        future_mass = weights[0, h].triu(diagonal=1).sum().item()
        ok = future_mass < 1e-5
        print(f"    Head {h}: future attention mass = {future_mass:.2e} → {'✓' if ok else '✗'}")

    params = sum(p.numel() for p in attn.parameters())
    print(f"\n  Total parameters in MHA: {params:,}")
    print(f"  (4 matrices of {d_model}×{d_model} = 4 × {d_model*d_model:,})")


if __name__ == "__main__":
    print("=" * 62)
    print("  ATTENTION MECHANISMS  —  the heart of the transformer")
    print("=" * 62)
    print("""
  At each position, attention decides: "which other positions
  matter most for understanding *me* right now?"

  It does this by:
    1. Projecting x into Q, K, V subspaces
    2. Scoring all (query, key) pairs with dot products
    3. Masking the future (causal), then normalizing with softmax
    4. Using the weights to mix the value vectors

  The result: each token becomes a weighted average of all the
  (past) tokens' values — it has "seen" its entire context.
""")

    demo_single_head()
    demo_multi_head()

    print("\n" + "=" * 62)
    print("  All attention tests passed. Ready for transformer_block.py")
    print("=" * 62 + "\n")
