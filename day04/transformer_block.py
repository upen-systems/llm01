"""
Day 04 — Transformer Block
===========================
Assembles all five components into one reusable GPT-style block:

  Pre-LayerNorm → Multi-Head Attention → Residual
  Pre-LayerNorm → Feed-Forward (MLP)   → Residual

Why "pre-norm" (LayerNorm BEFORE each sub-layer)?
  The original "Attention is All You Need" used post-norm.
  GPT-2 and later switched to pre-norm because it trains more
  stably — gradients flow cleanly through the residual path
  without passing through the norm operation.

Component recap from Claude.ai:
  LayerNorm   — normalizes each token's vector independently across its dims
                (mean=0, std=1, then learnable γ and β scale/shift)
  FFN / MLP   — two Linear layers with GELU in between
                expands d_model → 4×d_model → d_model
                lets each position "think" non-linearly after attention
  Residual    — x + sub_layer(x): keeps the original signal, adds a delta
                why: gradients skip straight to early layers → no vanishing

Stack multiple TransformerBlocks to build a full GPT.

Usage:
    python3 transformer_block.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Reuse the MHA we just built
sys.path.insert(0, os.path.dirname(__file__))
from attention import MultiHeadAttention


# ---------------------------------------------------------------------------
# Feed-Forward Network (Position-wise MLP)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Two-layer MLP applied identically to each token position.

    Architecture: Linear → GELU → Linear
    Hidden dim is 4× the model dim (standard in GPT).

    Why GELU instead of ReLU?
      GELU is smoother — it doesn't hard-zero negatives, it gently
      suppresses them proportional to how negative they are.
      Empirically outperforms ReLU in transformer-scale models.

    Why position-wise (independent per token)?
      Attention already mixed information across positions.
      The FFN then processes each position's enriched vector in isolation,
      acting as a per-token "memory" or "computation" step.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)    # expand: 32 → 128
        self.fc2 = nn.Linear(d_ff,   d_model)  # contract: 128 → 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = self.fc1(x)            # → (B, T, d_ff)
        x = F.gelu(x)              # smooth non-linearity
        x = self.fc2(x)            # → (B, T, d_model)
        return x


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    One GPT-style transformer layer.

    Forward pass:
        x = x + MHA(LayerNorm(x))     ← attention sub-layer, pre-norm, residual
        x = x + FFN(LayerNorm(x))     ← feed-forward sub-layer, pre-norm, residual

    The residual (x + ...) is what makes transformers trainable at depth:
    gradients flow straight through '+' without modification, so early
    layers still get a strong gradient signal even in 96-layer models.

    LayerNorm normalizes across the d_model dimension for each token
    independently — stabilizes activations so attention scores don't
    explode and GELU doesn't saturate.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()

        # Pre-norm before attention
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)

        # Pre-norm before feed-forward
        self.ln2 = nn.LayerNorm(d_model)
        self.ff  = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor):
        """
        x : (B, T, d_model)
        Returns:
          x       : (B, T, d_model) — same shape, enriched with context
          weights : (B, H, T, T)    — attention patterns from MHA
        """
        # Sub-layer 1: Multi-Head Self-Attention
        #   Normalize x first (pre-norm), then attend, then ADD back to x
        attn_out, weights = self.attn(self.ln1(x))
        x = x + attn_out                      # residual connection

        # Sub-layer 2: Feed-Forward Network
        #   Same pattern: normalize, transform, add back
        x = x + self.ff(self.ln2(x))          # residual connection

        return x, weights


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def print_section(title: str):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def demo_feedforward():
    print_section("FEED-FORWARD NETWORK (MLP)")

    d_model, d_ff = 32, 128
    B, T = 1, 5

    ff = FeedForward(d_model, d_ff)
    x  = torch.randn(B, T, d_model)

    with torch.no_grad():
        out = ff(x)

    print(f"\n  d_model={d_model}, d_ff={d_ff} (4× expansion)")
    print(f"\n  Input  shape : {list(x.shape)}")
    print(f"  Hidden shape : ({B}, {T}, {d_ff})   ← after fc1 + GELU")
    print(f"  Output shape : {list(out.shape)}")
    print(f"\n  Parameters   : {count_params(ff):,}")
    print(f"    fc1: {d_model}×{d_ff} = {d_model*d_ff:,} weights")
    print(f"    fc2: {d_ff}×{d_model} = {d_ff*d_model:,} weights")

    print("\n  GELU vs ReLU (first token, first 8 pre-activation values):")
    with torch.no_grad():
        h = ff.fc1(x[0, 0])[:8]
        relu_out = F.relu(h)
        gelu_out = F.gelu(h)
    print(f"\n  pre-act : {h.tolist()}")
    print(f"  ReLU    : {relu_out.tolist()}")
    print(f"  GELU    : {gelu_out.tolist()}")
    print("\n  Note: GELU gently suppresses negatives (not a hard zero)")


def demo_layernorm():
    print_section("LAYER NORMALIZATION")

    d_model = 32
    B, T    = 1, 4

    x  = torch.randn(B, T, d_model) * 5 + 3  # mean≈3, std≈5 (not normalized)
    ln = nn.LayerNorm(d_model)

    with torch.no_grad():
        out = ln(x)

    print(f"\n  Input  — mean: {x[0,0].mean():.3f}, std: {x[0,0].std():.3f}  (arbitrary scale)")
    print(f"  Output — mean: {out[0,0].mean():.3f}, std: {out[0,0].std():.3f}  (standardized)")
    print(f"\n  LayerNorm normalizes ACROSS d_model (per token, not per batch).")
    print(f"  Then applies learnable γ (scale) and β (shift), both init'd to 1 and 0.")
    print(f"\n  Why this matters for attention:")
    print(f"    QKᵀ / √d = dot product of two d-dim vectors.")
    print(f"    If vectors have large magnitudes, scores blow up → softmax → near-0/1")
    print(f"    → one head dominates, gradients vanish. LayerNorm prevents this.")


def demo_transformer_block():
    print_section("TRANSFORMER BLOCK — full assembly")

    d_model = 32
    n_heads = 8
    d_ff    = 128  # 4× d_model, standard GPT ratio
    B, T    = 2, 8

    block = TransformerBlock(d_model, n_heads, d_ff)
    x     = torch.randn(B, T, d_model)

    with torch.no_grad():
        out, weights = block(x)

    print(f"\n  Config: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
    print(f"\n  Input  shape   : {list(x.shape)}")
    print(f"  Output shape   : {list(out.shape)}     (identical — stack N blocks freely)")
    print(f"  Weights shape  : {list(weights.shape)}")

    print(f"\n  Parameter count breakdown:")
    print(f"    LayerNorm × 2 : {count_params(block.ln1) + count_params(block.ln2):>6,}")
    print(f"    MHA           : {count_params(block.attn):>6,}  (W_q, W_k, W_v, W_o)")
    print(f"    FeedForward   : {count_params(block.ff):>6,}  (fc1 + fc2)")
    print(f"    ─────────────────────────")
    print(f"    Total         : {count_params(block):>6,}")

    print(f"\n  Residual check — output should differ from input (not identity):")
    diff = (out - x).abs().mean().item()
    print(f"    Mean |out - x| = {diff:.4f}  (> 0 → block added something)")

    print(f"\n  Gradient flow check — can we backprop through the block?")
    x_grad = torch.randn(B, T, d_model, requires_grad=True)
    out2, _ = block(x_grad)
    loss = out2.sum()
    loss.backward()
    grad_norm = x_grad.grad.norm().item()
    print(f"    ∂loss/∂x norm = {grad_norm:.4f}  (> 0 → gradients flow cleanly)")
    assert grad_norm > 0, "No gradient!"
    print(f"    ✓ Backpropagation works")

    print(f"\n  Data flow diagram:")
    print(f"    x ({B},{T},{d_model})")
    print(f"    │")
    print(f"    ├─ LN1 → MHA → ──────────────────────────────── (+) → x'")
    print(f"    │                                                 ↑")
    print(f"    │                              residual skip ─────┘")
    print(f"    │")
    print(f"    ├─ LN2 → FFN → ──────────────────────────────── (+) → x''")
    print(f"    │                                                 ↑")
    print(f"    │                              residual skip ─────┘")
    print(f"    │")
    print(f"    └─ output ({B},{T},{d_model})  ← can feed into next TransformerBlock")


def demo_stacked_blocks():
    print_section("STACKING BLOCKS — simulate a mini-GPT")

    d_model = 32
    n_heads = 8
    d_ff    = 128
    n_layers = 4
    B, T    = 1, 6

    blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
    x = torch.randn(B, T, d_model)
    x_orig = x.clone()

    print(f"\n  {n_layers} transformer blocks, each {count_params(blocks[0]):,} params")
    print(f"  Total params: {count_params(blocks):,}")
    print(f"\n  Passing x through each block (tracking how much x changes):\n")
    print(f"  {'Block':<8} {'Mean |Δx|':>12}  {'Output norm':>14}")

    with torch.no_grad():
        for i, block in enumerate(blocks):
            x_before = x.clone()
            x, _ = block(x)
            delta = (x - x_before).abs().mean().item()
            norm  = x.norm().item()
            print(f"  Block {i+1:<4} {delta:>12.5f}  {norm:>14.5f}")

    total_change = (x - x_orig).abs().mean().item()
    print(f"\n  Total change from input to final output: {total_change:.5f}")
    print(f"\n  ✓ Each block transforms the representations.")
    print(f"    In a real GPT, the final x is then projected to vocab logits.")


if __name__ == "__main__":
    print("=" * 62)
    print("  TRANSFORMER BLOCK  —  assembling the full GPT building unit")
    print("=" * 62)
    print("""
  One transformer block = attention + FFN + normalization + residuals.
  Stack N of these to build GPT-2 (12), GPT-3 (96), or GPT-4 (~120+).

  The magic: because of residual connections, each block adds a small
  correction to x rather than replacing it. The original token embeddings
  "flow through" all layers and arrive at the output intact — each block
  just refines them incrementally.
""")

    demo_feedforward()
    demo_layernorm()
    demo_transformer_block()
    demo_stacked_blocks()

    print("\n" + "=" * 62)
    print("  All transformer block tests passed. Run test_day04.py next.")
    print("=" * 62 + "\n")
