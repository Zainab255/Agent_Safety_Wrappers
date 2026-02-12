"""Parallel Safety Orchestrator — the Agentic Control Layer.

Instantiates three cognitive perspectives (Red Team, Blue Team, Judge)
in a single batched forward pass and derives a continuous "Safety Entropy"
metric from the geometric disagreement of their last-token hidden-state
embeddings.

Decision logic:
    Entropy < sanitize_threshold  → ALLOW
    Entropy > block_threshold     → BLOCK
    Otherwise                     → REQUERY (sanitize with rigid template)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.llm_client import LLMClient
from wrappers.base import BaseWrapper, WrapperDecision, WrapperResult

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.json"

# Rigid sanitization template used when decision is REQUERY
SANITIZE_TEMPLATE = (
    "The following request has been flagged as ambiguous by the safety "
    "system. Please respond only with factual, educational information "
    "and refuse any harmful instructions.\n\n"
    "Original request (sanitized): {prompt}"
)


class SafetyOrchestrator(BaseWrapper):
    """Parallel multi-agent safety wrapper using embedding-space disagreement.

    Parameters
    ----------
    llm_client : LLMClient
        An already-loaded LLMClient instance (model in memory).
    config_path : str | None
        Path to config.json. Uses the repo default when *None*.
    """

    name = "safety_orchestrator"

    def __init__(
        self,
        llm_client: LLMClient,
        config_path: Optional[str] = None,
    ):
        config_file = Path(config_path) if config_path else CONFIG_PATH
        with open(config_file) as f:
            self.config = json.load(f)

        self.llm_client = llm_client
        self.personas: Dict[str, str] = self.config["agent_personas"]
        self.block_threshold: float = self.config["thresholds"]["entropy_block_threshold"]
        self.sanitize_threshold: float = self.config["thresholds"]["entropy_sanitize_threshold"]

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_batch(self, user_prompt: str) -> List[str]:
        """Combine each agent persona with the user prompt into a batch."""
        batch: List[str] = []
        for role in ("red_team", "blue_team", "judge"):
            system_prompt = self.personas[role]
            # Format: "<system>\n{persona}\n</system>\n\nUser: {prompt}"
            full_prompt = (
                f"<system>\n{system_prompt}\n</system>\n\nUser: {user_prompt}"
            )
            batch.append(full_prompt)
        return batch

    # ------------------------------------------------------------------
    # Entropy calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_safety_entropy(
        embeddings: torch.Tensor,
    ) -> Tuple[float, float]:
        """Compute Safety Entropy from the Red/Blue embedding pair.

        Args:
            embeddings: Tensor of shape ``[3, hidden_size]`` (red, blue, judge).

        Returns:
            (safety_entropy, red_blue_similarity)
        """
        red = embeddings[0]   # Red Team embedding
        blue = embeddings[1]  # Blue Team embedding

        similarity = F.cosine_similarity(
            red.unsqueeze(0), blue.unsqueeze(0)
        ).item()

        entropy = 1.0 - similarity
        return entropy, similarity

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _make_decision(
        self,
        entropy: float,
        user_prompt: str,
    ) -> Tuple[WrapperDecision, str, Optional[str]]:
        """Map entropy value to a wrapper decision.

        Returns:
            (decision, explanation, sanitized_prompt | None)
        """
        if entropy < self.sanitize_threshold:
            return (
                WrapperDecision.ALLOW,
                f"Low entropy ({entropy:.4f}) — agents agree prompt is safe.",
                None,
            )

        if entropy > self.block_threshold:
            return (
                WrapperDecision.BLOCK,
                f"High entropy ({entropy:.4f}) — agents strongly disagree; "
                "likely adversarial or jailbreak attempt.",
                None,
            )

        # Ambiguous zone → sanitize
        sanitized = SANITIZE_TEMPLATE.format(prompt=user_prompt)
        return (
            WrapperDecision.REQUERY,
            f"Medium entropy ({entropy:.4f}) — ambiguous prompt, sanitizing.",
            sanitized,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, prompt: str, **kwargs) -> WrapperResult:
        """Run the parallel safety orchestration pipeline.

        1. Build the 3-prompt batch (Red, Blue, Judge).
        2. Extract last-token embeddings in a single forward pass.
        3. Compute Safety Entropy (1 − cosine similarity of Red & Blue).
        4. Return a ``WrapperResult`` with decision and metrics.
        """
        batch = self._build_batch(prompt)
        logger.debug("Orchestrator batch built (%d prompts)", len(batch))

        embeddings = self.llm_client.get_agent_embeddings(batch)
        logger.debug("Embeddings shape: %s", embeddings.shape)

        entropy, similarity = self._compute_safety_entropy(embeddings)
        logger.info(
            "Safety Entropy=%.4f  Red-Blue Similarity=%.4f", entropy, similarity
        )

        decision, explanation, sanitized = self._make_decision(entropy, prompt)

        return WrapperResult(
            wrapper=self.name,
            decision=decision,
            explanation=explanation,
            sanitized_prompt=sanitized,
            metrics={
                "safety_entropy": round(entropy, 6),
                "red_blue_similarity": round(similarity, 6),
            },
        )
