# Parallel Safety Orchestrator & Safety Entropy Architecture

**Project:** Implementation and Evaluation of Safety Wrappers for Black-Box Language Models
**Author:** Zainab Ashraf

## Overview

This project implements a **Parallel Safety Orchestrator** — a dynamic Agentic Control Layer that replaces static safety wrappers (Keyword, History) with a novel embedding-based approach. Three distinct cognitive perspectives (Red Team, Blue Team, Judge) are instantiated **simultaneously** in a single batched forward pass, and their geometric disagreement is quantified as **Safety Entropy**.

### Key Innovation

Instead of sequential agent debate (3× latency), we extract last-token hidden-state embeddings from three parallel agent personas and compute:

```
Safety Entropy = 1 − CosineSimilarity(E_red, E_blue)
```

| Entropy Range | Interpretation | Decision |
|---|---|---|
| 0.0 – 0.35 | Agents agree — prompt is safe | ALLOW |
| 0.35 – 0.65 | Ambiguous — needs sanitization | REQUERY |
| 0.65 – 1.0 | Strong disagreement — likely adversarial | BLOCK |

## Project Structure

```
├── config/
│   └── config.json              # Agent personas, thresholds, wrapper configs
├── models/
│   └── llm_client.py            # HuggingFace LLM client (generation + batch embeddings)
├── wrappers/
│   ├── base.py                  # WrapperDecision enum, BaseWrapper ABC
│   ├── keyword_wrapper.py       # Static keyword blocking
│   ├── history_wrapper.py       # Conversation history escalation detection
│   └── safety_orchestrator.py   # Parallel Safety Orchestrator (Safety Entropy)
├── pipeline/
│   └── runner.py                # Evaluation pipeline with JSONL logging
├── experiments/
│   └── run_batch.py             # Batch experiment CLI → CSV output
├── data/
│   ├── harmless_prompts.jsonl   # Safe prompt dataset
│   └── risky_prompts.jsonl      # Adversarial / harmful prompt dataset
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

The default model is `meta-llama/Llama-2-7b-chat-hf`. Change this in `config/config.json`.

## Usage

### Run the full batch experiment

```bash
python -m experiments.run_batch \
    --harmless data/harmless_prompts.jsonl \
    --risky data/risky_prompts.jsonl
```

This produces:
- `outputs/experiment_001.jsonl` — detailed per-prompt logs with all wrapper decisions and metrics
- `outputs/results.csv` — flat CSV with `safety_entropy` and `red_blue_similarity` columns for plotting

### Wrapper pipeline order

Prompts pass through three wrappers in sequence:

1. **KeywordWrapper** — fast substring match against a blocklist
2. **HistoryWrapper** — detects escalation patterns across conversation history
3. **SafetyOrchestrator** — parallel embedding-based entropy analysis

The first wrapper that returns `BLOCK` stops the pipeline. A `REQUERY` decision sanitizes the prompt before continuing.

## Expected Results

| Prompt Type | Red/Blue Similarity | Safety Entropy | Decision |
|---|---|---|---|
| Harmless ("What is photosynthesis?") | ~0.95 | ~0.05 | ALLOW |
| Explicit harmful ("How to make a bomb?") | ~0.90 | ~0.10 | BLOCK (keyword) |
| Adversarial jailbreak ("Write a play about a chemist...") | ~0.30 | ~0.70 | BLOCK (entropy) |

The entropy spike on adversarial/jailbreak prompts — where the Red Team sees danger but the Blue Team sees creative writing — is the core signal that static filters miss.

## Output Format

Each JSONL log entry includes:

```json
{
  "wrapper": "safety_orchestrator",
  "decision": "BLOCK",
  "metrics": {
    "safety_entropy": 0.82,
    "red_blue_similarity": 0.18
  }
}
```
