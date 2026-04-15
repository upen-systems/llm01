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

