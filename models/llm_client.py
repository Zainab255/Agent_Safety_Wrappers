"""LLM Client with support for text generation and embedding extraction.

Provides a unified interface for loading a HuggingFace model and performing:
  - Standard text generation
  - Batch embedding extraction (last hidden state of the final token)
    used by the Parallel Safety Orchestrator
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.json"


class LLMClient:
    """Wraps a HuggingFace causal-LM for generation and gray-box embedding access."""

    def __init__(self, config_path: Optional[str] = None):
        config_file = Path(config_path) if config_path else CONFIG_PATH
        with open(config_file) as f:
            self.config = json.load(f)

        model_cfg = self.config["model"]
        self.model_name: str = model_cfg["name"]
        self.hf_token: Optional[str] = model_cfg.get("hf_token")
        self.device: str = model_cfg.get("device", "auto")
        self.max_new_tokens: int = model_cfg.get("max_new_tokens", 256)
        self.temperature: float = model_cfg.get("temperature", 0.7)

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> "LLMClient":
        """Load the tokenizer and model into memory."""
        token = self.hf_token if self.hf_token and self.hf_token != "YOUR_HF_TOKEN_HERE" else None

        logger.info("Loading tokenizer for %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left", token=token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading model %s (device=%s)", self.model_name, self.device)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            output_hidden_states=True,
            token=token,
        )
        self.model.eval()
        return self

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Generate a text completion for a single prompt."""
        assert self.model is not None and self.tokenizer is not None, (
            "Call .load() before generating."
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
            )
        # Decode only the newly generated tokens
        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Batch embedding extraction (gray-box signal)
    # ------------------------------------------------------------------

    def get_agent_embeddings(self, prompts: List[str]) -> torch.Tensor:
        """Run a batched forward pass and return last-token hidden states.

        Args:
            prompts: A list of N fully-formed prompt strings (one per agent
                     persona).

        Returns:
            Tensor of shape ``[N, hidden_size]`` — the last hidden-state
            vector of the final *non-padding* token for each prompt in the
            batch.
        """
        assert self.model is not None and self.tokenizer is not None, (
            "Call .load() before extracting embeddings."
        )

        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple of (num_layers+1) tensors,
        # each of shape [batch, seq_len, hidden_size].
        # We want the last layer's hidden states.
        last_hidden = outputs.hidden_states[-1]  # [N, seq_len, hidden_size]

        # For each item in the batch, pick the embedding of the last
        # *real* (non-padding) token using the attention mask.
        attention_mask = inputs["attention_mask"]  # [N, seq_len]
        # Index of the last non-pad token for each row
        seq_lengths = attention_mask.sum(dim=1) - 1  # [N]

        batch_size = last_hidden.size(0)
        embeddings = last_hidden[
            torch.arange(batch_size, device=last_hidden.device), seq_lengths
        ]  # [N, hidden_size]

        return embeddings
