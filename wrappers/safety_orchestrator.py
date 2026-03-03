"""Parallel Safety Orchestrator — the Agentic Control Layer.

Instantiates three cognitive perspectives (Red Team, Blue Team, Judge)
in a single batched forward pass and derives a continuous "Safety Entropy"
metric from the geometric disagreement of their L2-normalised last-token
hidden-state embeddings.

Embedding extraction details (for reproducibility):
    Layer:   Last transformer layer (``hidden_states[-1]``)
    Pooling: Last non-padding token per sequence (attention-mask indexed)
    Norm:    L2-normalised to unit length before cosine computation

Decision logic:
    Entropy < sanitize_threshold  → ALLOW
    Entropy > block_threshold     → BLOCK
    Otherwise                     → REQUERY (sanitize with rigid template)
"""

from __future__ import annotations

import json
import logging
import time
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
    sequential : bool
        If *True*, run the three persona prompts one at a time instead of
        as a single batch.  Used for the parallel-vs-sequential latency
        comparison experiment.
    """

    name = "safety_orchestrator"

    def __init__(
        self,
        llm_client: LLMClient,
        config_path: Optional[str] = None,
        sequential: bool = False,
    ):
        config_file = Path(config_path) if config_path else CONFIG_PATH
        with open(config_file) as f:
            self.config = json.load(f)

        self.llm_client = llm_client
        self.sequential = sequential
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
            full_prompt = (
                f"<system>\n{system_prompt}\n</system>\n\nUser: {user_prompt}"
            )
            batch.append(full_prompt)
        return batch

    # ------------------------------------------------------------------
    # Embedding retrieval (parallel vs sequential)
    # ------------------------------------------------------------------

    def _get_embeddings(self, batch: List[str]) -> Tuple[torch.Tensor, int]:
        """Retrieve embeddings, counting the number of model forward passes.

        Returns:
            (embeddings [3, hidden_size], model_calls)
        """
        if not self.sequential:
            # Single batched forward pass — the parallel path
            embeddings = self.llm_client.get_agent_embeddings(batch)
            return embeddings, 1

        # Sequential: one forward pass per persona
        parts = []
        for prompt in batch:
            emb = self.llm_client.get_agent_embeddings([prompt])
            parts.append(emb)
        embeddings = torch.cat(parts, dim=0)
        return embeddings, len(batch)

    # ------------------------------------------------------------------
    # Entropy calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_safety_entropy(
        embeddings: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute Safety Entropy from L2-normalised Red/Blue/Judge embeddings.

        Procedure (for paper reproducibility):
            1. Extract E_red (index 0), E_blue (index 1), E_judge (index 2).
            2. L2-normalise each vector to unit length.
            3. Compute pairwise cosine similarities.
            4. Safety Entropy = 1 − cos(E_red_norm, E_blue_norm).

        Args:
            embeddings: Tensor of shape ``[3, hidden_size]`` (red, blue, judge).

        Returns:
            Dict with keys: safety_entropy, red_blue_similarity,
            red_judge_similarity, blue_judge_similarity.
        """
        red = embeddings[0]
        blue = embeddings[1]
        judge = embeddings[2]

        # L2 normalisation to unit vectors
        red_norm = F.normalize(red, p=2, dim=0)
        blue_norm = F.normalize(blue, p=2, dim=0)
        judge_norm = F.normalize(judge, p=2, dim=0)

        # Pairwise cosine similarities (equivalent to dot product after L2 norm)
        rb_sim = torch.dot(red_norm, blue_norm).item()
        rj_sim = torch.dot(red_norm, judge_norm).item()
        bj_sim = torch.dot(blue_norm, judge_norm).item()

        entropy = 1.0 - rb_sim

        return {
            "safety_entropy": round(entropy, 6),
            "red_blue_similarity": round(rb_sim, 6),
            "red_judge_similarity": round(rj_sim, 6),
            "blue_judge_similarity": round(bj_sim, 6),
        }

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
        2. Extract last-token embeddings (parallel or sequential).
        3. L2-normalise and compute Safety Entropy.
        4. Return a ``WrapperResult`` with decision and full metrics.
        """
        batch = self._build_batch(prompt)
        logger.debug("Orchestrator batch built (%d prompts)", len(batch))

        t_start = time.perf_counter()
        embeddings, model_calls = self._get_embeddings(batch)
        latency_s = time.perf_counter() - t_start

        logger.debug("Embeddings shape: %s", embeddings.shape)

        metrics = self._compute_safety_entropy(embeddings)
        entropy = metrics["safety_entropy"]

        logger.info(
            "Safety Entropy=%.4f  Red-Blue Sim=%.4f  (mode=%s, %.3fs)",
            entropy,
            metrics["red_blue_similarity"],
            "sequential" if self.sequential else "parallel",
            latency_s,
        )

        decision, explanation, sanitized = self._make_decision(entropy, prompt)

        metrics.update({
            "mode": "sequential" if self.sequential else "parallel",
            "model_calls": model_calls,
            "latency_seconds": round(latency_s, 6),
        })

        return WrapperResult(
            wrapper=self.name,
            decision=decision,
            explanation=explanation,
            sanitized_prompt=sanitized,
            metrics=metrics,
        )
