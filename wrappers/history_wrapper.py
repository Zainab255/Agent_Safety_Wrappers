"""History-based safety wrapper.

Monitors conversation history for escalation patterns that indicate
jailbreak or prompt-injection attempts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from wrappers.base import BaseWrapper, WrapperDecision, WrapperResult

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.json"


class HistoryWrapper(BaseWrapper):
    """Detects escalation / jailbreak patterns across conversation history."""

    name = "history_wrapper"

    def __init__(self, config_path: Optional[str] = None):
        config_file = Path(config_path) if config_path else CONFIG_PATH
        with open(config_file) as f:
            cfg = json.load(f)

        hist_cfg = cfg.get("history_wrapper", {})
        self.max_history_length: int = hist_cfg.get("max_history_length", 10)
        self.escalation_patterns: List[str] = [
            p.lower() for p in hist_cfg.get("escalation_patterns", [])
        ]
        self._history: List[str] = []

    def add_to_history(self, prompt: str) -> None:
        """Append a prompt to the rolling history window."""
        self._history.append(prompt)
        if len(self._history) > self.max_history_length:
            self._history = self._history[-self.max_history_length :]

    def evaluate(self, prompt: str, **kwargs) -> WrapperResult:
        self.add_to_history(prompt)

        # Check the entire recent history for escalation patterns
        combined = " ".join(self._history).lower()
        matched = [p for p in self.escalation_patterns if p in combined]

        if matched:
            return WrapperResult(
                wrapper=self.name,
                decision=WrapperDecision.BLOCK,
                explanation=f"Escalation patterns detected in history: {matched}",
                metrics={
                    "matched_patterns": matched,
                    "history_length": len(self._history),
                },
            )

        return WrapperResult(
            wrapper=self.name,
            decision=WrapperDecision.ALLOW,
            explanation="No escalation patterns found.",
            metrics={
                "matched_patterns": [],
                "history_length": len(self._history),
            },
        )
