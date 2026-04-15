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
├── README.md              ← you are here
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

## Goal

Become an AI/LLM consultant who can advise companies on practical implementations — and back it up with hands-on credibility (including having built a transformer from scratch).














# LLM0.1 — 17-Day LLM Consultant Bootcamp

**Format:** 4 hrs/day · Learn in Claude.ai + Build in Claude Code
**Setup:** M1 MacBook Air 16GB · Claude.ai + Claude Code (subscription auth)
**Goal:** Become an AI/LLM consultant who can advise companies — and actually build one
**Project folder:** ~/llm01/

---

## Instructions for Claude (Claude.ai)

When the user uploads this file and says "LLM0.1":
1. Check which days have status [x] to know what's already done
2. Read the Log entries to understand what was built and learned
3. Start the next incomplete day
4. Each day has two tracks — run them in parallel:
   - **Claude.ai track:** Teach the concepts conversationally
   - **Claude Code track:** Give the user a project prompt to run in Claude Code in a separate terminal inside ~/llm01/
5. **At the end of each day, before closing:**
   - Use the memory_user_edits tool to save a summary of what was covered and built
   - Provide the user with an updated Log entry to paste into this file
   - Remind the user to mark status [x], save this file, and re-upload next session

## Instructions for Claude Code (via CLAUDE.md)

The user's ~/llm01/ project has a CLAUDE.md file that tells Claude Code:
- This is the LLM0.1 bootcamp project
- Each day's code goes in ~/llm01/dayXX/ folders (auto-created)
- After completing a build, update ~/llm01/progress.md with what was built
- The progress.md file is the shared log between Claude.ai and Claude Code

This way:
- Claude.ai reads this roadmap file to know the plan and learning progress
- Claude Code reads CLAUDE.md + progress.md to know what's been coded
- Both stay in sync through progress.md

---

## First-Time Setup

Run this in your terminal to create the project structure:

```bash
mkdir -p ~/llm01
cd ~/llm01

cat > CLAUDE.md << 'EOF'
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
EOF

cat > progress.md << 'EOF'
# LLM0.1 — Build Progress

This file is updated by Claude Code after each day's coding project.
Claude.ai reads this to stay in sync with what's been built.

---

EOF

echo "✅ LLM0.1 project created at ~/llm01/"
echo "📁 CLAUDE.md and progress.md are ready"
echo "🚀 cd ~/llm01 and run 'claude' to start Claude Code"
```

> **Note:** You do NOT need to create the day01/, day02/, etc. folders yourself.
> Claude Code will create them automatically when you start each day's build.

---

## Week 1: How LLMs Work + Build One

### Day 1: What Is an LLM, Really?
- **Goal:** Explain LLMs to a CEO in 60 seconds and a CTO in 5 minutes
- **Learn:** Next-token prediction, transformers/attention in plain English, pretraining → fine-tuning → RLHF, model landscape (Claude, GPT, Llama, Gemini, Mistral)
- **Build:** Token Explorer CLI — shows how tokenizers break text into tokens, with counts and cost estimates
- **Code folder:** ~/llm01/day01/
- **Status:** [x]
- **Log:**Completed Day 1. Claude.ai track: covered next-token prediction, 
tokenization (with hands-on Token Explorer), attention mechanism 
(horizontal token correlation + vertical layer refinement), 
deterministic vs probabilistic behavior, temperature, hallucination, 
and the core consulting question: "how deterministic are your 
requirements?" Build track: Token Explorer CLI built in day01/ using 
tiktoken — tokenizes input, shows token IDs, decoded strings, stats, 
and cost estimates across 3 models. Key insight: same meaning costs 
2x more tokens in Japanese vs English; clean code costs more tokens 
than terse code.

### Day 2: Context Windows & Model Limits
- **Goal:** Understand practical constraints every LLM deployment faces
- **Learn:** Context windows (what 200K tokens means), why LLMs hallucinate, temperature/top-p/top-k, input vs output token costs
- **Build:** LLM Cost Calculator Web App — helps companies estimate API costs by model
- **Code folder:** ~/llm01/day02/
- **Status:** [ ]
- **Log:**

### Day 3: Build a Tokenizer & Embeddings from Scratch
- **Goal:** Understand how raw text becomes numbers an LLM can process
- **Learn:** Byte Pair Encoding algorithm step by step, how vocabulary is built, token → integer → embedding vector pipeline, positional encodings (how the model knows word order)
- **Build:** BPE Tokenizer + Embedding Trainer — build a working BPE tokenizer from scratch (no libraries), train simple word embeddings on a text corpus, visualize the embedding space
- **Code folder:** ~/llm01/day03/
- **Status:** [ ]
- **Log:**

### Day 4: Build a Transformer — Attention Is All You Need
- **Goal:** Understand the transformer architecture by building every piece
- **Learn:** Self-attention (Q, K, V matrices — what they actually do), multi-head attention and why multiple heads help, layer norm, residual connections, feed-forward blocks, causal masking (how the model only looks backwards)
- **Build:** Transformer Block from Scratch — implement single attention head, scale to multi-head attention, build a complete GPT-style transformer block in PyTorch, test it processes data correctly
- **Code folder:** ~/llm01/day04/
- **Status:** [ ]
- **Log:**

### Day 5: Train Your Mini-LLM
- **Goal:** Train a working language model and generate text from it
- **Learn:** Data pipelines (cleaning, chunking, batching), training loops (loss functions, optimizers, learning rate schedules), evaluation (perplexity — what it means), decoding strategies (greedy, top-k, top-p, temperature — now you'll FEEL the difference)
- **Build:** Mini-GPT — assemble Day 3 tokenizer + Day 4 transformer into a complete model (~2-10M parameters), train on a text corpus (Shakespeare, Wikipedia subset, or your own data), build a text generation CLI with sampling controls, plot training loss curves
- **Code folder:** ~/llm01/day05/
- **Status:** [ ]
- **Log:**

### Day 6: The Model Landscape
- **Goal:** Know every major model and when to recommend each
- **Learn:** Open vs closed models, model sizes and tradeoffs, multimodal capabilities, benchmarks (MMLU, HumanEval, Arena ELO)
- **Build:** AI Model Comparison Dashboard — interactive comparison with recommendation engine
- **Code folder:** ~/llm01/day06/
- **Status:** [ ]
- **Log:**

### Day 7: Prompt Engineering Mastery
- **Goal:** Master reliable prompting patterns and build a testing framework
- **Learn:** System prompts, few-shot, chain-of-thought, structured outputs, prompt injection attacks/defenses, evaluation methods
- **Build:** Prompt Testing Lab — test prompts across parameters, template library, red-team mode
- **Code folder:** ~/llm01/day07/
- **Status:** [ ]
- **Log:**

---

## Week 2: Applied LLM Skills

### Day 8: RAG: The Enterprise Pattern
- **Goal:** Understand RAG deeply and build a working prototype
- **Learn:** RAG architecture (embed → store → retrieve → generate), vector databases, chunking strategies, when RAG fails
- **Build:** Chat With Your Documents — RAG app using ChromaDB + Claude
- **Code folder:** ~/llm01/day08/
- **Status:** [ ]
- **Log:**

### Day 9: Fine-Tuning & When NOT To
- **Goal:** Know the fine-tuning decision framework and run one locally
- **Learn:** Fine-tuning vs prompting vs RAG decision framework, LoRA/QLoRA, training costs, failure modes
- **Build:** Local Fine-Tune on M1 — LoRA fine-tune SmolLM-135M using MLX
- **Code folder:** ~/llm01/day09/
- **Status:** [ ]
- **Log:**

### Day 10: Agents, Tools & MCP
- **Goal:** Understand agents and build a working multi-tool agent
- **Learn:** Agent architectures (ReAct, plan-and-execute), tool use/function calling, MCP protocol, agent limitations
- **Build:** Research Agent — searches web, extracts facts, writes structured research briefs
- **Code folder:** ~/llm01/day10/
- **Status:** [ ]
- **Log:**

### Day 11: Safety, Security & Governance
- **Goal:** Be the person who raises risks before they become problems
- **Learn:** Prompt injection/jailbreaks, PII handling, EU AI Act, responsible AI frameworks
- **Build:** LLM Security Scanner — tests deployments for 20+ vulnerability types, generates risk report
- **Code folder:** ~/llm01/day11/
- **Status:** [ ]
- **Log:**

### Day 12: Running Local Models
- **Goal:** Know local model reality from first-hand experience
- **Learn:** Ollama, MLX, llama.cpp, quantization (Q4/Q8/GGUF), M1 performance reality, local vs cloud costs
- **Build:** Local Model Benchmark Suite — install/run/benchmark 4 models, generate comparison report
- **Code folder:** ~/llm01/day12/
- **Status:** [ ]
- **Log:**

---

## Week 3: Business & Capstone

### Day 13: API Integration Patterns
- **Goal:** Understand how engineering teams deploy LLMs in production
- **Learn:** Streaming, retries, rate limiting, fallbacks, function calling, caching, cost optimization
- **Build:** Production-Ready API Wrapper — robust client with caching, cost tracking, streaming, fallbacks
- **Code folder:** ~/llm01/day13/
- **Status:** [ ]
- **Log:**

### Day 14: Use Cases & ROI Frameworks
- **Goal:** Map LLM capabilities to business outcomes with real numbers
- **Learn:** High-ROI use cases by industry, build vs buy analysis, measuring success, why POCs fail
- **Build:** AI Opportunity Assessment Tool — guided assessment that generates ROI analysis and roadmap
- **Code folder:** ~/llm01/day14/
- **Status:** [ ]
- **Log:**

### Day 15: Evaluation & Quality Systems
- **Goal:** Know how to measure if an LLM solution is working
- **Learn:** Eval frameworks, LLM-as-judge, human eval design, production monitoring
- **Build:** LLM Evaluation Framework — reusable test runner with multiple scoring methods and reports
- **Code folder:** ~/llm01/day15/
- **Status:** [ ]
- **Log:**

### Day 16: Your Consulting Toolkit
- **Goal:** Package everything into a professional consulting practice
- **Learn:** Engagement structure (discovery → assessment → roadmap), pricing models, building credibility, staying current
- **Build:** Consultant Portfolio Site — professional website with services, case studies, blog
- **Code folder:** ~/llm01/day16/
- **Status:** [ ]
- **Log:**

### Day 17: Capstone: Mock Engagement
- **Goal:** Run a complete consulting engagement end-to-end
- **Learn:** Synthesize everything, present to different audiences, handle objections
- **Build:** Full Client Deliverable — AI Strategy Assessment doc + presentation deck + project charter
- **Code folder:** ~/llm01/day17/
- **Status:** [ ]
- **Log:**
