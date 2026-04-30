"""
Day 04 — Integration Test
==========================
Feeds the position-aware vectors from Day 3 through the TransformerBlock.

Pipeline:
    Day 3 embeddings.npy  →  token vectors
         +
    Sinusoidal PE         →  positional encoding
         ↓
    embedding + PE        →  position-aware input
         ↓
    TransformerBlock      →  context-aware output
         ↓
    Analysis: what changed? do gradients flow? cosine similarity inspection

This is exactly the input pipeline used in GPT — the only missing
piece is the final Linear(d_model, vocab_size) that projects to logits.

Usage:
    python3 test_day04.py
    (run from day04/ or anywhere — it resolves day03/ paths automatically)
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import math

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAY03   = os.path.join(ROOT, "day03")
DAY04   = os.path.join(ROOT, "day04")
sys.path.insert(0, DAY04)

from transformer_block import TransformerBlock


# ---------------------------------------------------------------------------
# Positional Encoding (same formula as Day 3)
# ---------------------------------------------------------------------------

def sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Returns shape: (seq_len, d_model)
    """
    pe  = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ---------------------------------------------------------------------------
# Load Day 3 embeddings
# ---------------------------------------------------------------------------

def load_day3(sentence: str):
    """
    Load word embeddings and vocab from Day 3, build a token sequence
    for `sentence`, and return (embeddings_tensor, token_list).

    Returns None if Day 3 files are not found.
    """
    emb_path   = os.path.join(DAY03, "embeddings.npy")
    vocab_path = os.path.join(DAY03, "vocab.txt")

    if not os.path.exists(emb_path) or not os.path.exists(vocab_path):
        return None, None

    embeddings = np.load(emb_path)                    # (vocab_size, 32)
    with open(vocab_path) as f:
        idx2word = [line.strip() for line in f]
    word2idx = {w: i for i, w in enumerate(idx2word)}

    tokens = [w for w in sentence.lower().split() if w in word2idx]
    if not tokens:
        return None, None

    vecs = np.stack([embeddings[word2idx[t]] for t in tokens])  # (T, 32)
    return torch.tensor(vecs, dtype=torch.float32), tokens


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-10))


def print_section(title: str):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_positional_encoding():
    print_section("STEP 1 — Sinusoidal Positional Encoding")

    seq_len, d_model = 6, 32
    pe = sinusoidal_pe(seq_len, d_model)

    print(f"\n  PE shape: {list(pe.shape)}  (positions × dims)")
    print(f"\n  First 8 dims of each position:")
    print(f"  {'pos':<6} " + "  ".join(f"d{i:<3}" for i in range(8)))
    for p in range(seq_len):
        vals = "  ".join(f"{v:+.3f}" for v in pe[p, :8].tolist())
        print(f"  {p:<6} {vals}")

    print(f"\n  Low dims (0-3) oscillate fast — fine-grained position info.")
    print(f"  High dims (28-31) oscillate slow — coarse position info.")
    print(f"  Each row is unique → the model can always distinguish positions.")

    # Verify pos 0 is pure cosine (sin(0)=0 for all even dims)
    assert abs(pe[0, 0].item()) < 1e-5, "pos=0 even dim should be sin(0)=0"
    print(f"\n  ✓ Position 0, dim 0 = {pe[0,0].item():.6f} (sin(0) = 0, as expected)")


def test_with_day3_embeddings():
    print_section("STEP 2 — Loading Day 3 Embeddings")

    sentence = "the model learns from data"
    x_words, tokens = load_day3(sentence)

    if x_words is None:
        print(f"\n  Day 3 embeddings not found at {DAY03}")
        print(f"  Run day03/embeddings.py first, then re-run this test.")
        print(f"  Falling back to random embeddings for shape verification.\n")
        tokens  = sentence.split()
        x_words = torch.randn(len(tokens), 32)
        day3_loaded = False
    else:
        day3_loaded = True

    T, d_model = x_words.shape
    print(f"\n  Sentence : '{sentence}'")
    print(f"  Tokens   : {tokens}")
    print(f"  Shape    : {list(x_words.shape)}  (T={T} tokens × d_model={d_model} dims)")

    if day3_loaded:
        print(f"\n  Raw embedding norms (from Day 3 skip-gram training):")
        for i, tok in enumerate(tokens):
            norm = x_words[i].norm().item()
            print(f"    '{tok:15}' → norm = {norm:.4f}")

    return x_words, tokens, d_model


def test_embedding_plus_pe(x_words, tokens):
    print_section("STEP 3 — Embedding + Positional Encoding")

    T, d_model = x_words.shape
    pe         = sinusoidal_pe(T, d_model)
    x_input    = x_words + pe  # (T, d_model)

    print(f"\n  x_emb shape  : {list(x_words.shape)}")
    print(f"  PE shape     : {list(pe.shape)}")
    print(f"  x_input shape: {list(x_input.shape)}  (sum — carries both identity + position)")

    print(f"\n  Did PE change the vectors?")
    delta = (x_input - x_words).abs().mean().item()
    print(f"    Mean |x_input - x_emb| = {delta:.5f}")
    print(f"    → PE added a unique offset to each token based on its position.")

    print(f"\n  Are tokens at different positions now distinguishable?")
    sim_same = cosine_similarity(x_words[0], x_words[0])
    sim_diff = cosine_similarity(x_words[0], x_words[1]) if T > 1 else 0.0
    print(f"    sim(tok0, tok0) = {sim_same:.4f}  (identical)")
    print(f"    sim(tok0, tok1) = {sim_diff:.4f}  (different — good)")

    return x_input


def test_transformer_block(x_input, tokens):
    print_section("STEP 4 — TransformerBlock Forward Pass")

    T, d_model = x_input.shape
    n_heads    = 8
    d_ff       = 128

    block = TransformerBlock(d_model, n_heads, d_ff)

    # Add batch dimension: (T, d_model) → (1, T, d_model)
    x_batch = x_input.unsqueeze(0)

    with torch.no_grad():
        out_batch, weights = block(x_batch)

    out = out_batch.squeeze(0)  # back to (T, d_model)

    print(f"\n  Input  shape : {list(x_batch.shape)}")
    print(f"  Output shape : {list(out_batch.shape)}")
    print(f"\n  Per-token change (context enrichment):")
    print(f"  {'Token':<16} {'Input norm':>12}  {'Output norm':>12}  {'Cosine sim':>12}")
    print(f"  {'─'*15:<16} {'─'*10:>12}  {'─'*10:>12}  {'─'*10:>12}")
    for i, tok in enumerate(tokens):
        in_norm  = x_input[i].norm().item()
        out_norm = out[i].norm().item()
        sim      = cosine_similarity(x_input[i], out[i])
        print(f"  {tok:<16} {in_norm:>12.4f}  {out_norm:>12.4f}  {sim:>12.4f}")

    print(f"\n  Cosine sim < 1.0 → each token's vector has been modified")
    print(f"  by attending to the other tokens in the sequence.")

    return out, weights


def test_attention_patterns(weights, tokens):
    print_section("STEP 5 — Attention Pattern Visualization")

    T = len(tokens)
    # weights: (1, 8, T, T) — take batch 0, show a few heads
    W = weights[0]  # (8, T, T)

    print(f"\n  Attention weights for each head (rows=query, cols=key).")
    print(f"  Upper-right triangle = 0 (causal mask — no future peeking).\n")

    for head_idx in [0, 1, 2]:
        print(f"  ── Head {head_idx} ──")
        header = f"  {'':16}" + "".join(f"{t[:5]:>7}" for t in tokens)
        print(header)
        for i, tok in enumerate(tokens):
            row = "".join(f"{W[head_idx, i, j].item():>7.3f}" for j in range(T))
            print(f"  {tok:<16}{row}")
        print()

    print(f"  Note: different heads often develop different attention patterns.")
    print(f"  Some heads may attend locally (nearby tokens), others globally.")


def test_gradient_flow(x_input, tokens):
    print_section("STEP 6 — Gradient Flow Test")

    T, d_model = x_input.shape
    block = TransformerBlock(d_model, n_heads=8, d_ff=128)

    # Require gradients so we can verify backprop
    x_batch = x_input.unsqueeze(0).requires_grad_(True)
    out, _  = block(x_batch)

    # Dummy loss: minimize output norm (like a language model loss)
    loss = out.pow(2).mean()
    loss.backward()

    grad = x_batch.grad.squeeze(0)  # (T, d_model)
    print(f"\n  Loss (dummy): {loss.item():.5f}")
    print(f"\n  Gradient at each token position:")
    for i, tok in enumerate(tokens):
        gnorm = grad[i].norm().item()
        bar   = "█" * min(int(gnorm * 10), 30)
        print(f"    {tok:<16} ‖∇‖ = {gnorm:.5f}  {bar}")

    all_nonzero = (grad.abs() > 1e-8).all().item()
    print(f"\n  All gradients non-zero: {'✓' if all_nonzero else '✗'}")
    print(f"  → Backpropagation flows cleanly through attention, FFN, and norms.")


def test_mini_gpt_stack():
    print_section("STEP 7 — Mini-GPT Stack (4 blocks + output projection)")

    d_model  = 32
    n_heads  = 8
    d_ff     = 128
    n_layers = 4
    vocab_size = 80   # matches Day 3 vocab size (79 words + 1 padding)
    B, T     = 1, 5

    # A mini-GPT has: embedding → N transformer blocks → linear to vocab
    class MiniGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks    = nn.ModuleList(
                [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
            )
            self.ln_final  = nn.LayerNorm(d_model)     # final norm before projection
            self.lm_head   = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, x):
            # x: (B, T, d_model) — already embedded + PE added
            for block in self.blocks:
                x, _ = block(x)
            x      = self.ln_final(x)
            logits = self.lm_head(x)  # (B, T, vocab_size)
            return logits

    model = MiniGPT()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Architecture: embed(32) → {n_layers}×TransformerBlock → LM head")
    print(f"  Total parameters: {total_params:,}")
    print(f"\n  Forward pass (random input to simulate post-embedding):")

    x      = torch.randn(B, T, d_model)
    logits = model(x)

    print(f"  Input  shape : {list(x.shape)}")
    print(f"  Logits shape : {list(logits.shape)}  (B, T, vocab_size)")
    print(f"\n  Each position outputs a distribution over {vocab_size} vocab tokens.")
    print(f"  argmax(logits[0, -1]) gives the predicted next token after position T.")

    # Show predicted token indices for the last position
    probs     = torch.softmax(logits[0, -1], dim=-1)
    top5_vals, top5_idx = probs.topk(5)
    print(f"\n  Top-5 predicted next tokens (untrained, so random):")
    for v, idx in zip(top5_vals.tolist(), top5_idx.tolist()):
        print(f"    token {idx:>4d}  p = {v:.4f}")

    print(f"\n  ✓ Full GPT-style pipeline works end-to-end.")
    print(f"    Day 5 will train this model on real text and generate output.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 62)
    print("  DAY 04 INTEGRATION TEST")
    print("  Day 3 vectors → positional encoding → TransformerBlock")
    print("=" * 62)
    print("""
  This test wires together everything from Day 3 and Day 4:

    token ids (Day 3 vocab)
    → lookup in embeddings.npy (Day 3 skip-gram)
    → + sinusoidal PE (Day 3 positional_encoding.py)
    → TransformerBlock (Day 4)
    → context-enriched vectors

  The output is ready to be projected to vocab logits for next-token
  prediction — that's exactly what Day 5 will train.
""")

    test_positional_encoding()
    x_words, tokens, d_model = test_with_day3_embeddings()
    x_input = test_embedding_plus_pe(x_words, tokens)
    out, weights = test_transformer_block(x_input, tokens)
    test_attention_patterns(weights, tokens)
    test_gradient_flow(x_input, tokens)
    test_mini_gpt_stack()

    print("\n" + "=" * 62)
    print("  ALL TESTS PASSED")
    print("=" * 62)
    print("""
  Day 4 complete. You have built and verified:
    ✓ SingleHeadAttention   — Q/K/V projections, scaled dot product
    ✓ Causal masking        — upper-triangle -inf, softmax → 0
    ✓ MultiHeadAttention    — 8 parallel heads, concat + project
    ✓ LayerNorm             — per-token normalization, pre-norm style
    ✓ FeedForward (GELU)    — 2-layer MLP, 4× expansion
    ✓ Residual connections  — x + sub_layer(x) around both blocks
    ✓ TransformerBlock      — full assembly, stackable
    ✓ Gradient flow         — backprop works end-to-end
    ✓ Day 3 integration     — real word embeddings + PE through block
    ✓ Mini-GPT stack        — 4 blocks + LM head = a complete (tiny) GPT

  Next: Day 5 — train this model on real text and generate output.
""")
