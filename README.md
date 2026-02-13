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

## Where Inference-Time Guidance & Latent-Space Analysis Are Used

This project combines two techniques — **inference-time guidance** (steering model behavior at runtime through prompt construction) and **latent-space analysis** (inspecting internal representations rather than generated text) — across several components:

### Inference-Time Guidance

Inference-time guidance steers model behavior without fine-tuning by injecting role-specific system prompts at the moment of inference.

| File | What happens |
|---|---|
| `config/config.json` | Defines three agent personas — each is a system-level instruction that reframes how the model processes the same user input at inference time. |
| `wrappers/safety_orchestrator.py` → `_build_batch()` | Constructs the actual guided prompts. Each user input is paired with a different persona (`<system>` block), producing three distinct inference-time framings from a single model: **Paranoid** (Red Team), **Helpful** (Blue Team), and **Constitutional** (Judge). |
| `wrappers/safety_orchestrator.py` → `SANITIZE_TEMPLATE` | When the decision is REQUERY, a rigid sanitization template wraps the original prompt — another form of inference-time guidance that constrains the model's downstream generation to factual, educational responses only. |
| `pipeline/runner.py` → `evaluate_prompt()` | Applies the sanitized prompt to the generation call when the orchestrator returns REQUERY, closing the loop between guidance and output. |

### Latent-Space Analysis

Latent-space analysis treats the model as a gray box — we read its internal hidden states rather than relying solely on generated text.

| File | What happens |
|---|---|
| `models/llm_client.py` → `get_agent_embeddings()` | The core gray-box extraction point. Runs a batched forward pass with `output_hidden_states=True`, then extracts `outputs.hidden_states[-1][:, -1, :]` — the **last layer, last token** embedding for each prompt in the batch. This yields a `[3, hidden_size]` tensor of latent representations. |
| `models/llm_client.py` → attention mask indexing | Uses the attention mask to find the true final non-padding token per sequence, ensuring the latent vector is meaningful even with variable-length inputs in the batch. |
| `wrappers/safety_orchestrator.py` → `_compute_safety_entropy()` | Operates entirely in latent space. Computes `cosine_similarity(E_red, E_blue)` between the Red Team and Blue Team embedding vectors, then derives `Safety Entropy = 1 − similarity`. No text is generated or parsed — the safety signal comes purely from geometric disagreement in the model's internal representation space. |
| `wrappers/safety_orchestrator.py` → `_make_decision()` | Maps the continuous latent-space metric (entropy) to a discrete safety decision using configurable thresholds. |

### How They Work Together

```
User Prompt
    │
    ├──→ [Inference-Time Guidance]  3 persona-framed prompts built
    │         │
    │         ▼
    │    [Latent-Space Analysis]   Single batched forward pass → 3 embeddings
    │         │
    │         ▼
    │    Cosine Similarity (Red vs Blue)  →  Safety Entropy
    │         │
    │         ├── Low entropy   → ALLOW
    │         ├── Mid entropy   → REQUERY  ──→ [Inference-Time Guidance] sanitized template
    │         └── High entropy  → BLOCK
    │
    ▼
  Final Decision + Metrics logged
```

The key insight is that neither technique is sufficient alone. Inference-time guidance creates the divergent perspectives, and latent-space analysis quantifies their disagreement without the cost and noise of generating and parsing text from three agents.

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
