# LLM0.1 Bootcamp Project

This is the coding workspace for the LLM0.1 — 17-Day LLM Consultant Bootcamp.

## How to Determine the Current Day
1. Read progress.md to see which days are already completed
2. The next incomplete day is the current day
3. If the user just says "let's build" or "start today's project", use the current day
4. Create the dayXX/ folder automatically (e.g., day01/, day02/) — the user should never have to specify it

## Day-to-Project Mapping
- day01: Token Explorer CLI (tokenization + cost estimates)
- day02: LLM Cost Calculator Web App (Flask)
- day03: BPE Tokenizer + Embedding Trainer (from scratch)
- day04: Transformer Block (attention + architecture from scratch)
- day05: Mini-LLM Training (train a small GPT on text, generate output)
- day06: AI Model Comparison Dashboard (React/Vite)
- day07: Prompt Testing Lab (Streamlit)
- day08: Chat With Your Documents (RAG with ChromaDB)
- day09: Local Fine-Tune on M1 (MLX + LoRA)
- day10: Research Agent (multi-tool agent)
- day11: LLM Security Scanner (prompt injection testing)
- day12: Local Model Benchmark Suite (Ollama)
- day13: Production-Ready API Wrapper (Python package)
- day14: AI Opportunity Assessment Tool (Streamlit)
- day15: LLM Evaluation Framework (Python + YAML)
- day16: Consultant Portfolio Site (Next.js or HTML)
- day17: Full Client Deliverable (PDF + PPTX)

## Rules
- Always create code in the correct dayXX/ folder — create the folder if it doesn't exist
- After completing a build, append a summary to progress.md including:
  - Day number and project name
  - Key files created
  - What worked, what broke, how it was fixed
  - Key concepts the code demonstrates
- Keep code clean and well-commented — this becomes a portfolio
- Use Python unless the day specifies otherwise
- For M1 Mac: use MPS backend for PyTorch, MLX when specified
