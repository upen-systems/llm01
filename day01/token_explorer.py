#!/usr/bin/env python3
"""
Day 01 — Token Explorer CLI
Tokenizes text using tiktoken (cl100k_base) and shows token IDs,
decoded strings, counts, char/token ratio, and estimated API costs.
"""

import sys
import argparse
import tiktoken

# ANSI color codes for cycling through token colors
COLORS = [
    "\033[91m",  # red
    "\033[92m",  # green
    "\033[93m",  # yellow
    "\033[94m",  # blue
    "\033[95m",  # magenta
    "\033[96m",  # cyan
    "\033[97m",  # white
    "\033[33m",  # dark yellow
    "\033[36m",  # dark cyan
    "\033[35m",  # dark magenta
]
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Pricing per 1M input tokens
MODELS = {
    "GPT-4o":        5.00,
    "Claude Sonnet": 3.00,
    "Llama (local)": 0.00,
}


def colorize(text: str, color_index: int) -> str:
    return f"{COLORS[color_index % len(COLORS)]}{text}{RESET}"


def analyze(text: str) -> None:
    enc = tiktoken.get_encoding("cl100k_base")
    token_ids = enc.encode(text)
    n_tokens = len(token_ids)
    n_chars = len(text)
    ratio = n_chars / n_tokens if n_tokens else 0

    # ── Header ───────────────────────────────────────────────────────────────
    print(f"\n{BOLD}── Token Explorer ──────────────────────────────────{RESET}")
    print(f"  Encoding : cl100k_base  (GPT-4 / Claude approximation)")

    # ── Colorized token display ───────────────────────────────────────────────
    print(f"\n{BOLD}Tokens (color-coded boundaries):{RESET}")
    colored_pieces = []
    for i, tid in enumerate(token_ids):
        piece = enc.decode([tid])
        colored_pieces.append(colorize(piece, i))
    # Join with a thin separator so boundaries are obvious even on same color runs
    print("  " + "·".join(colored_pieces))

    # ── Token table ───────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'#':<6} {'Token ID':<12} {'Decoded'}{RESET}")
    print(f"  {'─'*4}   {'─'*10}   {'─'*20}")
    for i, tid in enumerate(token_ids):
        piece = enc.decode([tid])
        display = repr(piece)          # repr so whitespace/newlines are visible
        colored = colorize(display, i)
        print(f"  {i:<4}   {tid:<10}   {colored}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}── Stats ───────────────────────────────────────────{RESET}")
    print(f"  Token count   : {BOLD}{n_tokens}{RESET}")
    print(f"  Char count    : {n_chars}")
    print(f"  Chars / token : {ratio:.2f}")

    # ── Cost estimates ────────────────────────────────────────────────────────
    print(f"\n{BOLD}── Estimated API cost (input only) ─────────────────{RESET}")
    for model, price_per_m in MODELS.items():
        cost = (n_tokens / 1_000_000) * price_per_m
        if price_per_m == 0.0:
            cost_str = f"{DIM}$0.00  (local — free){RESET}"
        else:
            cost_str = f"${cost:.6f}  ({price_per_m:.2f}/1M tokens)"
        print(f"  {model:<18}: {cost_str}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize text and show cost estimates.",
        epilog='Example: python day01/token_explorer.py "Hello, world!"',
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to tokenize. If omitted, reads from stdin or prompts.",
    )
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif not sys.stdin.isatty():
        # piped input: echo "hello" | python token_explorer.py
        text = sys.stdin.read().rstrip("\n")
    else:
        # interactive fallback
        print("Enter text to tokenize (Ctrl+D to finish):")
        try:
            text = sys.stdin.read().rstrip("\n")
        except KeyboardInterrupt:
            sys.exit(0)

    if not text:
        print("No input provided.", file=sys.stderr)
        sys.exit(1)

    analyze(text)


if __name__ == "__main__":
    main()
