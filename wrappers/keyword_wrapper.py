"""Static keyword-based safety wrapper.

Blocks prompts that contain any of the configured blocked keywords
(case-insensitive substring match).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from wrappers.base import BaseWrapper, WrapperDecision, WrapperResult

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.json"


class KeywordWrapper(BaseWrapper):
    """Rejects prompts that contain explicitly blocked keywords."""

    name = "keyword_wrapper"

    def __init__(self, config_path: Optional[str] = None):
        config_file = Path(config_path) if config_path else CONFIG_PATH
        with open(config_file) as f:
            cfg = json.load(f)
        self.blocked_keywords: List[str] = [
            kw.lower() for kw in cfg.get("keyword_wrapper", {}).get("blocked_keywords", [])
        ]

    def evaluate(self, prompt: str, **kwargs) -> WrapperResult:
        prompt_lower = prompt.lower()
        matched = [kw for kw in self.blocked_keywords if kw in prompt_lower]

        if matched:
            return WrapperResult(
                wrapper=self.name,
                decision=WrapperDecision.BLOCK,
                explanation=f"Blocked keywords detected: {matched}",
                metrics={"matched_keywords": matched},
            )

        return WrapperResult(
            wrapper=self.name,
            decision=WrapperDecision.ALLOW,
            explanation="No blocked keywords found.",
            metrics={"matched_keywords": []},
        )
