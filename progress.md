# LLM0.1 — Build Progress

This file is updated by Claude Code after each day's coding project.
Claude.ai reads this to stay in sync with what's been built.

---

## Day 01 — Token Explorer CLI

**Project:** Token Explorer CLI
**Folder:** `day01/token_explorer.py`

### Key Files
- `day01/token_explorer.py` — single-file CLI tool (~120 lines)

### What Was Built
A terminal tool that tokenizes any text using tiktoken's `cl100k_base` encoding (the same encoder used by GPT-4 and a close approximation for Claude). It accepts a CLI argument, piped stdin, or falls back to an interactive prompt.

**Output includes:**
- Color-coded token boundaries displayed inline (10 cycling ANSI colors)
- A table of token index → integer ID → decoded string (repr'd so whitespace is visible)
- Token count, character count, chars-per-token ratio
- Estimated input cost for GPT-4o ($5/1M), Claude Sonnet ($3/1M), and Llama local ($0)

### What Worked
- `tiktoken` installed cleanly via pip3 and `cl100k_base` loaded without issue
- ANSI color cycling makes token boundaries immediately obvious even for short texts
- Both `python3 day01/token_explorer.py "text"` and `echo "text" | python3 day01/token_explorer.py` work correctly

### Nothing Broke

### Key Concepts Demonstrated
- **BPE tokenization** — words split at subword boundaries; leading spaces are part of tokens
- **Token ≠ word** — punctuation is its own token; "The" vs " the" are different token IDs
- **Cost math** — `(n_tokens / 1_000_000) * price_per_million`
- **Chars/token ratio** — typical English prose ≈ 4 chars/token; code and special chars lower this

---

## Day 02 — LLM Cost Calculator Web App

**Project:** LLM Cost Calculator Web App
**Folder:** `day02/app.py`

### Key Files
- `day02/app.py` — single-file Flask app (~280 lines, inline CSS template)

### What Was Built
A single-page Flask web app that estimates LLM API costs for a given use case. Runs on port 5002.

**Features:**
- Input form: use case description, monthly requests, avg input/output tokens, model selector
- 6 models with accurate $/1M pricing: Claude Sonnet 4, Claude Haiku 4.5, GPT-4o, GPT-4o mini, Gemini 1.5 Pro, Gemini 1.5 Flash
- Results: monthly cost, annual cost, cost per request, input vs output breakdown bar chart
- All-model comparison table sorted by monthly cost with provider badges
- Context window warning when avg input tokens > 50,000
- Prompt caching section: checkbox enables cached token field; shows monthly savings + % reduction (Anthropic models only, 90% discount on cached tokens)

### What Worked
- Flask installed cleanly via pip3; app served 200 on first run
- POST calculation verified correct: cost math, caching discount, comparison sort all pass
- Dark UI rendered cleanly with inline ANSI-free CSS — no external deps

### Nothing Broke

### Key Concepts Demonstrated
- **Token pricing math** — `(tokens / 1_000_000) * price_per_million` applied to input and output separately
- **Prompt caching economics** — 90% discount on repeated system prompt tokens dramatically reduces Anthropic costs at scale
- **Input vs output asymmetry** — output tokens are 3–5× more expensive than input across all providers
- **Provider cost spread** — Gemini Flash is ~40× cheaper than Claude Sonnet for the same workload
- **Flask GET/POST pattern** — single route handles both form render and result display

---

## Day 03 — BPE Tokenizer + Embedding Trainer

**Project:** BPE Tokenizer + Embedding Trainer (from scratch)
**Folder:** `day03/`

### Key Files
- `day03/bpe_tokenizer.py` — BPE algorithm from scratch, encode/decode, merge history
- `day03/embeddings.py` — skip-gram word2vec in PyTorch, MPS backend, saves embeddings.npy
- `day03/visualize.py` — PCA projection to 2D, matplotlib scatter plot, nearest neighbors
- `day03/positional_encoding.py` — sinusoidal PE, heatmap + waveform plot

### What Was Built
Four standalone scripts that together cover the full text → numbers → meaning pipeline:

1. **BPE Tokenizer** — trains on a 20-sentence corpus, learns 54 merge rules to hit vocab size 80, prints each merge step with frequencies. `encode()` returns integer token IDs; `decode()` reconstructs original text. Showed BPE merging "token" from t+o+k+e+n in 5 steps.

2. **Embedding Trainer** — skip-gram PyTorch model with negative sampling loss, trained 50 epochs on MPS (M1 GPU). Saves 79×32 embedding matrix to `embeddings.npy`. Nearest neighbors show semantically related words clustering (attention → tokens, scores, vectors, layers).

3. **Visualizer** — loads embeddings, runs PCA to 2D, plots labelled scatter with dark theme. Prints top-5 cosine-similarity neighbors for model, token, training, attention, embedding.

4. **Positional Encoding** — implements PE(pos, 2i) = sin / cos formula from "Attention Is All You Need". Generates 50×64 matrix, saves heatmap showing low-dim fast oscillation vs high-dim slow oscillation.

### What Worked
- BPE correctly merges high-frequency pairs first (s+</w> → "s</w>" at step 1 with freq=25)
- MPS backend activated automatically on M1 — training ran on GPU
- encode/decode round-trip verified on all test sentences
- PE position 0 correctly produces all-cosine encoding (sin(0)=0)

### Nothing Broke

### Key Concepts Demonstrated
- **BPE algorithm** — starts with characters, merges most-frequent pairs iteratively; "token" emerges naturally from the corpus
- **Subword tokenization** — rare words decompose into known subpieces; `embeddings` → `e m b e d d ing s</w>`
- **Skip-gram objective** — predict context words from center word; similar-context words get similar vectors
- **Negative sampling** — train binary classifiers (real pair vs noise) instead of softmax over full vocab
- **MPS acceleration** — PyTorch `.to("mps")` runs on M1 GPU with no code changes beyond device detection
- **Sinusoidal PE** — unique fingerprint per position; low dims oscillate fast (fine-grained), high dims slow (coarse)

---

## Day 04 — Transformer Block

**Project:** GPT-style Transformer Block (from scratch)
**Folder:** `day04/`

### Key Files
- `day04/attention.py` — SingleHeadAttention + MultiHeadAttention (8 heads), causal masking
- `day04/transformer_block.py` — FeedForward (GELU), LayerNorm, residuals, TransformerBlock, stacking demo
- `day04/test_day04.py` — 7-step integration test using Day 3 embeddings + PE

### What Was Built
Three files that build a complete GPT-style transformer block from the ground up:

1. **SingleHeadAttention** — One Q/K/V projection each. Explicit causal mask via `torch.triu(..., diagonal=1)` filled with `-inf` before softmax. Printed the full 6×6 attention weight matrix to verify upper-triangle is exactly 0.

2. **MultiHeadAttention** — 8 heads, each operating on 4-dim subspace (32/8=4). Single big W_q/W_k/W_v projections reshaped into `(B, H, T, d_head)` for efficiency. Output projection W_o recombines heads. All 8 heads verified: row-sums == 1, future mass == 0.

3. **FeedForward** — Linear(32→128) + GELU + Linear(128→32). Printed GELU vs ReLU side-by-side: GELU gently suppresses negatives (not hard-zero). 4× expansion = 8,352 parameters.

4. **LayerNorm** — Verified normalization: input mean=2.19/std=4.26 → output mean=0.00/std=1.02. Explained why pre-norm prevents attention score explosion.

5. **TransformerBlock** — Pre-norm variant: `x = x + MHA(LN(x))`, then `x = x + FFN(LN(x))`. 12,576 total params per block. Gradient norm = 24.4, confirms backprop works.

6. **Stacked blocks** — 4 blocks, 50,304 total params. Tracked mean |Δx| per layer (~0.22–0.28) showing each block incrementally refines representations.

7. **Integration test** — Loaded `day03/embeddings.npy` (79×32) + `vocab.txt`, selected "the model learns from data", added sinusoidal PE, ran through TransformerBlock. Cosine sims 0.87–0.93 confirmed vectors were enriched. Gradient flows to all 5 token positions.

8. **Mini-GPT** — 4 TransformerBlocks + final LayerNorm + Linear(32, 80) = 52,928 params. Produces logits shape (1, 5, 80). Verified argmax gives a predicted next token.

### What Worked
- Causal mask via `torch.triu` + `masked_fill(-inf)` — clean, verified mathematically
- Multi-head reshape trick `view(B, T, H, d_head).transpose(1,2)` — no separate head loops needed
- Day 3 embeddings loaded directly into the pipeline — real word vectors through real attention
- Gradient flow confirmed via `.requires_grad_(True)` + `loss.backward()` on all inputs

### Nothing Broke

### Key Concepts Demonstrated
- **Scaled dot-product attention** — `QKᵀ / √d_k` prevents softmax saturation at high dims
- **Causal masking** — GPT can only look backward; BERT can look both ways; mask enforces this
- **Multi-head attention** — 8 parallel subspaces each learn a different relationship pattern
- **Pre-norm vs post-norm** — pre-norm (GPT-2+) trains more stably; gradients flow clean through residual
- **Residual connections** — `x + sub_layer(x)` means gradients skip layers; no vanishing at 96 layers
- **Feed-forward as token memory** — attention mixes positions; FFN processes each position independently
- **GELU** — smooth non-linearity; outperforms ReLU in transformer-scale models
- **Stackable design** — input shape == output shape, so N blocks chain without adapters
