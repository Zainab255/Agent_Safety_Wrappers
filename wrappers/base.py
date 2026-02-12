"""Base classes and enums shared by all safety wrappers."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class WrapperDecision(enum.Enum):
    """Outcome of a safety wrapper evaluation."""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REQUERY = "REQUERY"  # sanitize and re-submit with a rigid template


@dataclass
class WrapperResult:
    """Structured result returned by every wrapper's ``evaluate`` method."""

    wrapper: str
    decision: WrapperDecision
    explanation: str = ""
    sanitized_prompt: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class BaseWrapper:
    """Abstract base class for all safety wrappers.

    Subclasses must implement ``evaluate(prompt, **kwargs) -> WrapperResult``.
    """

    name: str = "base"

    def evaluate(self, prompt: str, **kwargs) -> WrapperResult:
        raise NotImplementedError
