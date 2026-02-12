#!/usr/bin/env python3
"""Batch experiment runner.

Loads prompts from JSONL data files, evaluates them through the full
safety pipeline (Keyword → History → SafetyOrchestrator), and writes
structured results to both JSONL (for replay) and CSV (for analysis /
plotting in the paper).

CSV columns include ``safety_entropy`` and ``red_blue_similarity`` so
that Entropy Distribution graphs can be produced directly.

Usage:
    python -m experiments.run_batch \
        --harmless data/harmless_prompts.jsonl \
        --risky    data/risky_prompts.jsonl \
        [--config  config/config.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.llm_client import LLMClient
from pipeline.runner import PipelineRunner
from wrappers.keyword_wrapper import KeywordWrapper
from wrappers.history_wrapper import HistoryWrapper
from wrappers.safety_orchestrator import SafetyOrchestrator

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_prompts(path: str) -> List[Dict]:
    """Load prompts from a JSONL file.

    Each line should be a JSON object with at least a ``"prompt"`` key.
    An optional ``"category"`` key is preserved for analysis.
    """
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


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
    )
    parser.add_argument(
        "--harmless",
        default=str(PROJECT_ROOT / "data" / "harmless_prompts.jsonl"),
        help="Path to harmless prompts JSONL.",
    )
    parser.add_argument(
        "--risky",
        default=str(PROJECT_ROOT / "data" / "risky_prompts.jsonl"),
        help="Path to risky prompts JSONL.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config.json"),
        help="Path to config.json.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

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

    for label, path in [("harmless", args.harmless), ("risky", args.risky)]:
        if not Path(path).exists():
            logger.warning("Skipping %s — file not found: %s", label, path)
            continue
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


if __name__ == "__main__":
    main()
