#!/usr/bin/env python3
<<<<<<< HEAD
"""Batch experiment runner.

Loads prompts from JSONL data files, evaluates them through the full
safety pipeline (Keyword → History → SafetyOrchestrator), and writes
structured results to both JSONL (for replay) and CSV (for analysis /
plotting in the paper).

CSV columns include ``safety_entropy`` and ``red_blue_similarity`` so
that Entropy Distribution graphs can be produced directly.
=======
"""Batch experiment runner with repeatability, baselines, and parallel-vs-
sequential comparison.

Runs every prompt through multiple method pipelines over N independent runs
(each with a logged random seed), then computes aggregate statistics
(mean, std, 95 % confidence interval) and writes both per-run and summary
CSV files for direct use in the paper.

Method pipelines evaluated:
    1. keyword + history + orchestrator (parallel)   — proposed method
    2. keyword + history + orchestrator (sequential)  — latency comparison
    3. keyword + history + llm_judge                  — strong baseline A
    4. keyword + history + self_critique              — strong baseline B
>>>>>>> fd982a5 (Formalized Safety Entropy and strengthened evaluation framework)

Usage:
    python -m experiments.run_batch \
        --harmless data/harmless_prompts.jsonl \
        --risky    data/risky_prompts.jsonl \
<<<<<<< HEAD
        [--config  config/config.json]
=======
        [--config  config/config.json] \
        [--runs 5] [--seed 42]
>>>>>>> fd982a5 (Formalized Safety Entropy and strengthened evaluation framework)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
<<<<<<< HEAD
import sys
from pathlib import Path
from typing import Dict, List, Optional
=======
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
>>>>>>> fd982a5 (Formalized Safety Entropy and strengthened evaluation framework)

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.llm_client import LLMClient
from pipeline.runner import PipelineRunner
from wrappers.keyword_wrapper import KeywordWrapper
from wrappers.history_wrapper import HistoryWrapper
from wrappers.safety_orchestrator import SafetyOrchestrator
<<<<<<< HEAD
=======
from wrappers.llm_judge_wrapper import LLMJudgeWrapper
from wrappers.self_critique_wrapper import SelfCritiqueWrapper
>>>>>>> fd982a5 (Formalized Safety Entropy and strengthened evaluation framework)

logger = logging.getLogger(__name__)


<<<<<<< HEAD
# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_prompts(path: str) -> List[Dict]:
    """Load prompts from a JSONL file.

    Each line should be a JSON object with at least a ``"prompt"`` key.
    An optional ``"category"`` key is preserved for analysis.
    """
=======
# ======================================================================
# Helpers
# ======================================================================

def load_prompts(path: str) -> List[Dict]:
    """Load prompts from a JSONL file."""
>>>>>>> fd982a5 (Formalized Safety Entropy and strengthened evaluation framework)
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


<<<<<<< HEAD
def extract_orchestrator_metrics(record: Dict) -> Dict[str, Optional[float]]:
    """Pull safety_entropy and red_blue_similarity out of a pipeline record."""
    for wr in record.get("wrapper_results", []):
        if wr.get("wrapper") == "safety_orchestrator":
            metrics = wr.get("metrics", {})
            return {
                "safety_entropy": metrics.get("safety_entropy"),
                "red_blue_similarity": metrics.get("red_blue_similarity"),
            }
    return {"safety_entropy": None, "red_blue_similarity": None}


def records_to_csv(records: List[Dict], output_path: str) -> None:
    """Write experiment records to a flat CSV for plotting."""
    fieldnames = [
        "prompt",
        "category",
        "final_decision",
        "safety_entropy",
        "red_blue_similarity",
        "keyword_decision",
        "history_decision",
        "orchestrator_decision",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in records:
            orch_metrics = extract_orchestrator_metrics(rec)

            # Per-wrapper decisions
            wrapper_decisions: Dict[str, str] = {}
            for wr in rec.get("wrapper_results", []):
                wrapper_decisions[wr["wrapper"]] = wr["decision"]

            writer.writerow(
                {
                    "prompt": rec["prompt"],
                    "category": rec.get("category", "unknown"),
                    "final_decision": rec["final_decision"],
                    "safety_entropy": orch_metrics["safety_entropy"],
                    "red_blue_similarity": orch_metrics["red_blue_similarity"],
                    "keyword_decision": wrapper_decisions.get("keyword_wrapper", ""),
                    "history_decision": wrapper_decisions.get("history_wrapper", ""),
                    "orchestrator_decision": wrapper_decisions.get(
                        "safety_orchestrator", ""
                    ),
                }
            )

    logger.info("CSV written to %s", output_path)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the safety-wrapper batch experiment."
=======
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_wrapper_metric(
    record: Dict, wrapper_name: str, metric_key: str
) -> Optional[Any]:
    """Pull a specific metric from a named wrapper inside a pipeline record."""
    for wr in record.get("wrapper_results", []):
        if wr.get("wrapper") == wrapper_name:
            return wr.get("metrics", {}).get(metric_key)
    return None


def extract_wrapper_decision(record: Dict, wrapper_name: str) -> str:
    """Pull the decision string from a named wrapper."""
    for wr in record.get("wrapper_results", []):
        if wr.get("wrapper") == wrapper_name:
            return wr.get("decision", "")
    return ""


# ======================================================================
# CSV writers
# ======================================================================

PER_RUN_FIELDS = [
    "run_id",
    "seed",
    "pipeline",
    "prompt",
    "category",
    "final_decision",
    "safety_entropy",
    "red_blue_similarity",
    "red_judge_similarity",
    "blue_judge_similarity",
    "keyword_decision",
    "history_decision",
    "orchestrator_decision",
    "llm_judge_decision",
    "self_critique_decision",
    "total_latency_seconds",
    "total_model_calls",
]

STATS_FIELDS = [
    "pipeline",
    "category",
    "metric",
    "mean",
    "std",
    "ci95_lower",
    "ci95_upper",
    "n",
]


def records_to_csv(records: List[Dict], output_path: str) -> None:
    """Write per-run experiment records to a flat CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_RUN_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "run_id": rec.get("run_id", 0),
                "seed": rec.get("seed", 0),
                "pipeline": rec.get("pipeline", ""),
                "prompt": rec["prompt"],
                "category": rec.get("category", "unknown"),
                "final_decision": rec["final_decision"],
                "safety_entropy": extract_wrapper_metric(
                    rec, "safety_orchestrator", "safety_entropy"
                ),
                "red_blue_similarity": extract_wrapper_metric(
                    rec, "safety_orchestrator", "red_blue_similarity"
                ),
                "red_judge_similarity": extract_wrapper_metric(
                    rec, "safety_orchestrator", "red_judge_similarity"
                ),
                "blue_judge_similarity": extract_wrapper_metric(
                    rec, "safety_orchestrator", "blue_judge_similarity"
                ),
                "keyword_decision": extract_wrapper_decision(rec, "keyword_wrapper"),
                "history_decision": extract_wrapper_decision(rec, "history_wrapper"),
                "orchestrator_decision": extract_wrapper_decision(
                    rec, "safety_orchestrator"
                ),
                "llm_judge_decision": extract_wrapper_decision(rec, "llm_judge"),
                "self_critique_decision": extract_wrapper_decision(rec, "self_critique"),
                "total_latency_seconds": rec.get("total_latency_seconds"),
                "total_model_calls": rec.get("total_model_calls"),
            })
    logger.info("Per-run CSV written to %s", output_path)


def compute_aggregate_stats(
    records: List[Dict], stats_csv: str
) -> None:
    """Compute mean, std, and 95 % CI per pipeline × category × metric."""
    from collections import defaultdict

    # Bucket: (pipeline, category, metric_name) → [values]
    buckets: Dict[tuple, List[float]] = defaultdict(list)

    for rec in records:
        key_base = (rec.get("pipeline", ""), rec.get("category", "unknown"))

        # Latency
        lat = rec.get("total_latency_seconds")
        if lat is not None:
            buckets[(*key_base, "latency_seconds")].append(float(lat))

        # Model calls
        mc = rec.get("total_model_calls")
        if mc is not None:
            buckets[(*key_base, "model_calls")].append(float(mc))

        # Safety entropy (only for orchestrator pipelines)
        ent = extract_wrapper_metric(rec, "safety_orchestrator", "safety_entropy")
        if ent is not None:
            buckets[(*key_base, "safety_entropy")].append(float(ent))

        # Red-blue similarity
        rbs = extract_wrapper_metric(rec, "safety_orchestrator", "red_blue_similarity")
        if rbs is not None:
            buckets[(*key_base, "red_blue_similarity")].append(float(rbs))

        # False-positive flag: BLOCK on harmless
        if rec.get("category") in ("harmless", "science", "technology", "math",
                                     "literature", "geography", "health"):
            is_fp = 1.0 if rec.get("final_decision") == "BLOCK" else 0.0
            buckets[(*key_base, "false_positive_rate")].append(is_fp)

        # True-positive flag: BLOCK on risky
        if rec.get("category") in ("explicit_harmful", "adversarial_jailbreak",
                                     "prompt_injection", "risky"):
            is_tp = 1.0 if rec.get("final_decision") == "BLOCK" else 0.0
            buckets[(*key_base, "true_positive_rate")].append(is_tp)

    with open(stats_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATS_FIELDS)
        writer.writeheader()
        for (pipeline, category, metric), values in sorted(buckets.items()):
            n = len(values)
            mean = sum(values) / n
            std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5 if n > 1 else 0.0
            se = std / math.sqrt(n) if n > 1 else 0.0
            ci95 = 1.96 * se
            writer.writerow({
                "pipeline": pipeline,
                "category": category,
                "metric": metric,
                "mean": round(mean, 6),
                "std": round(std, 6),
                "ci95_lower": round(mean - ci95, 6),
                "ci95_upper": round(mean + ci95, 6),
                "n": n,
            })

    logger.info("Aggregate stats CSV written to %s", stats_csv)


# ======================================================================
# Pipeline builders
# ======================================================================

def build_pipelines(
    client: LLMClient, config_path: str
) -> Dict[str, PipelineRunner]:
    """Construct all method pipelines for the experiment."""
    keyword_w = KeywordWrapper(config_path=config_path)

    pipelines: Dict[str, PipelineRunner] = {}

    # 1. Proposed method: parallel orchestrator
    history_p = HistoryWrapper(config_path=config_path)
    orch_parallel = SafetyOrchestrator(
        llm_client=client, config_path=config_path, sequential=False
    )
    pipelines["parallel_orchestrator"] = PipelineRunner(
        wrappers=[keyword_w, history_p, orch_parallel],
        llm_client=client,
        config_path=config_path,
        pipeline_label="parallel_orchestrator",
    )

    # 2. Sequential orchestrator (latency comparison)
    history_s = HistoryWrapper(config_path=config_path)
    orch_sequential = SafetyOrchestrator(
        llm_client=client, config_path=config_path, sequential=True
    )
    pipelines["sequential_orchestrator"] = PipelineRunner(
        wrappers=[keyword_w, history_s, orch_sequential],
        llm_client=client,
        config_path=config_path,
        pipeline_label="sequential_orchestrator",
    )

    # 3. LLM-as-a-Judge baseline
    history_j = HistoryWrapper(config_path=config_path)
    judge_w = LLMJudgeWrapper(llm_client=client, config_path=config_path)
    pipelines["llm_judge"] = PipelineRunner(
        wrappers=[keyword_w, history_j, judge_w],
        llm_client=client,
        config_path=config_path,
        pipeline_label="llm_judge",
    )

    # 4. Self-Critique baseline
    history_c = HistoryWrapper(config_path=config_path)
    critique_w = SelfCritiqueWrapper(llm_client=client, config_path=config_path)
    pipelines["self_critique"] = PipelineRunner(
        wrappers=[keyword_w, history_c, critique_w],
        llm_client=client,
        config_path=config_path,
        pipeline_label="self_critique",
    )

    return pipelines


# ======================================================================
# Main
# ======================================================================

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the full safety-wrapper experiment suite."
>>>>>>> fd982a5 (Formalized Safety Entropy and strengthened evaluation framework)
    )
    parser.add_argument(
        "--harmless",
        default=str(PROJECT_ROOT / "data" / "harmless_prompts.jsonl"),
<<<<<<< HEAD
        help="Path to harmless prompts JSONL.",
=======
>>>>>>> fd982a5 (Formalized Safety Entropy and strengthened evaluation framework)
    )
    parser.add_argument(
        "--risky",
        default=str(PROJECT_ROOT / "data" / "risky_prompts.jsonl"),
<<<<<<< HEAD
        help="Path to risky prompts JSONL.",
=======
>>>>>>> fd982a5 (Formalized Safety Entropy and strengthened evaluation framework)
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config.json"),
<<<<<<< HEAD
        help="Path to config.json.",
    )
=======
    )
    parser.add_argument("--runs", type=int, default=None, help="Number of runs.")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed.")
>>>>>>> fd982a5 (Formalized Safety Entropy and strengthened evaluation framework)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

<<<<<<< HEAD
    # 1. Load config-driven values
    with open(args.config) as f:
        config = json.load(f)
    output_csv = config.get("experiment", {}).get(
        "output_csv", "outputs/results.csv"
    )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    # 2. Load LLM
    logger.info("Loading LLM client …")
    client = LLMClient(config_path=args.config).load()

    # 3. Build wrapper stack
    keyword_w = KeywordWrapper(config_path=args.config)
    history_w = HistoryWrapper(config_path=args.config)
    orchestrator_w = SafetyOrchestrator(llm_client=client, config_path=args.config)

    wrappers = [keyword_w, history_w, orchestrator_w]

    # 4. Build pipeline runner
    runner = PipelineRunner(
        wrappers=wrappers, llm_client=client, config_path=args.config
    )

    # 5. Load prompts
    all_records: List[Dict] = []

=======
    with open(args.config) as f:
        config = json.load(f)

    exp_cfg = config.get("experiment", {})
    num_runs = args.runs or exp_cfg.get("num_runs", 5)
    base_seed = args.seed if args.seed is not None else exp_cfg.get("random_seed", 42)
    output_csv = exp_cfg.get("output_csv", "outputs/results.csv")
    stats_csv = exp_cfg.get("stats_csv", "outputs/aggregate_stats.csv")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    # Load LLM
    logger.info("Loading LLM client …")
    client = LLMClient(config_path=args.config).load()

    # Build all pipelines
    pipelines = build_pipelines(client, args.config)

    # Load prompts
    prompt_sets: List[tuple] = []
>>>>>>> fd982a5 (Formalized Safety Entropy and strengthened evaluation framework)
    for label, path in [("harmless", args.harmless), ("risky", args.risky)]:
        if not Path(path).exists():
            logger.warning("Skipping %s — file not found: %s", label, path)
            continue
<<<<<<< HEAD
        prompt_data = load_prompts(path)
        logger.info("Loaded %d %s prompts from %s", len(prompt_data), label, path)

        raw_prompts = [p["prompt"] for p in prompt_data]
        records = runner.run_and_log(raw_prompts)

        # Annotate with category for the CSV
        for rec, src in zip(records, prompt_data):
            rec["category"] = src.get("category", label)

        all_records.extend(records)

    # 6. Write CSV
    if all_records:
        records_to_csv(all_records, output_csv)
        logger.info(
            "Experiment complete — %d prompts evaluated. CSV at %s",
            len(all_records),
            output_csv,
        )
    else:
        logger.warning("No prompts were evaluated. Check your data paths.")
=======
        data = load_prompts(path)
        logger.info("Loaded %d %s prompts from %s", len(data), label, path)
        prompt_sets.append((label, data))

    all_records: List[Dict] = []

    for run_idx in range(num_runs):
        run_seed = base_seed + run_idx
        set_seed(run_seed)
        logger.info("=== Run %d/%d  (seed=%d) ===", run_idx + 1, num_runs, run_seed)

        for pipe_name, runner in pipelines.items():
            for label, prompt_data in prompt_sets:
                raw_prompts = [p["prompt"] for p in prompt_data]
                records = runner.run_and_log(
                    raw_prompts, run_id=run_idx, seed=run_seed
                )
                for rec, src in zip(records, prompt_data):
                    rec["category"] = src.get("category", label)
                all_records.extend(records)

    # Write outputs
    if all_records:
        records_to_csv(all_records, output_csv)
        compute_aggregate_stats(all_records, stats_csv)
        logger.info(
            "Experiment complete — %d total records across %d runs. "
            "CSV: %s  Stats: %s",
            len(all_records), num_runs, output_csv, stats_csv,
        )
    else:
        logger.warning("No prompts were evaluated.")
>>>>>>> fd982a5 (Formalized Safety Entropy and strengthened evaluation framework)


if __name__ == "__main__":
    main()
