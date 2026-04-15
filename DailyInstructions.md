This file first has the daily instructions, and then the roadmap info.

# LLM0.1 — 17-Day LLM Consultant Bootcamp

A hands-on sprint to learn LLMs deeply by building one from scratch and shipping 17 consultant-grade projects in 17 days.

**Author:** Upen ([@upen-systems](https://github.com/upen-systems))
**Hardware:** M1 MacBook Air 16GB
**Tools:** Claude.ai (learning) + Claude Code (building)

---

## What's in this repo

Each day produces a working project that builds toward becoming an AI/LLM consultant. Days 3–5 build a transformer from scratch; the rest cover the applied skills consultants need.

| Day | Project | Status |
|-----|---------|--------|
| 01 | Token Explorer CLI | ✅ |
| 02 | LLM Cost Calculator Web App | ⬜ |
| 03 | BPE Tokenizer + Embeddings (from scratch) | ⬜ |
| 04 | Transformer Block (from scratch) | ⬜ |
| 05 | Mini-GPT Training | ⬜ |
| 06 | AI Model Comparison Dashboard | ⬜ |
| 07 | Prompt Testing Lab | ⬜ |
| 08 | Chat With Your Documents (RAG) | ⬜ |
| 09 | Local Fine-Tune on M1 (LoRA) | ⬜ |
| 10 | Research Agent | ⬜ |
| 11 | LLM Security Scanner | ⬜ |
| 12 | Local Model Benchmark Suite | ⬜ |
| 13 | Production-Ready API Wrapper | ⬜ |
| 14 | AI Opportunity Assessment Tool | ⬜ |
| 15 | LLM Evaluation Framework | ⬜ |
| 16 | Consultant Portfolio Site | ⬜ |
| 17 | Capstone: Full Client Deliverable | ⬜ |

See [`LLM0.1-roadmap.md`](./LLM0.1-roadmap.md) for the full plan and progress log.

---

## Repo structure

```
~/llm01/
├── DailyInstructions.md   ← you are here
├── README.md              ← copy of the original DailyInstructions and README  
├── LLM0.1-roadmap.md      ← master plan + daily logs
├── CLAUDE.md              ← instructions for Claude Code
├── progress.md            ← shared log between Claude.ai and Claude Code
├── day01/                 ← Token Explorer CLI
├── day02/                 ← (auto-created when you start the day)
└── ...
```

---

## Daily workflow

### Starting a session

**For the learning side (Claude.ai):**
1. Open a new Claude.ai chat
2. Upload `LLM0.1-roadmap.md`
3. Say `LLM0.1 — Day X` (or just `LLM0.1` to resume from the last incomplete day)

**For the building side (Claude Code):**
```bash
cd ~/llm01
claude
```
Then say `let's build today's project` — Claude Code reads `progress.md` and figures out which day it is.

### Ending a session — commit to GitHub

After finishing the day's work, run these three commands manually (good practice for learning Git):

```bash
cd ~/llm01
git add .
git commit -m "Day X: <project name>"
git push
```

Example:
```bash
git commit -m "Day 2: LLM Cost Calculator Web App"
```

### Common Git commands

| Command | What it does |
|---------|--------------|
| `git status` | See what's changed and what's staged |
| `git add .` | Stage all changes for commit |
| `git commit -m "message"` | Save staged changes locally with a message |
| `git push` | Send local commits up to GitHub |
| `git pull` | Pull latest changes from GitHub |
| `git log --oneline` | See commit history |
| `git diff` | See what changed in unstaged files |

---
