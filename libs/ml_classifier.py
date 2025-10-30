from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock


logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[\w']+")


def _tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token]


class ModelNotAvailableError(RuntimeError):
    """Raised when the ML model cannot be loaded."""


@dataclass
class LeadModelConfig:
    model_path: Path
    threshold: float

    @classmethod
    def from_env(cls) -> "LeadModelConfig":
        raw_path = os.getenv("LEAD_MODEL_PATH", "model_store/lead_classifier.json")
        raw_threshold = os.getenv("LEAD_MODEL_THRESHOLD", "0.5")
        try:
            threshold = float(raw_threshold)
        except ValueError:
            threshold = 0.5
        return cls(model_path=Path(raw_path), threshold=threshold)


class LeadMLClassifier:
    """Wrapper around the trained ML model used for lead detection."""

    def __init__(self, config: LeadModelConfig | None = None) -> None:
        self.config = config or LeadModelConfig.from_env()
        self._model: dict | None = None
        self._lock = Lock()

    # ------------------------------------------------------------------
    def _load_model(self) -> dict:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                if not self.config.model_path.exists():
                    raise ModelNotAvailableError(
                        f"Model not found at {self.config.model_path}"
                    )
                logger.info("Loading ML lead classifier from %s", self.config.model_path)
                raw = self.config.model_path.read_text(encoding="utf-8")
                self._model = json.loads(raw)
        return self._model

    # ------------------------------------------------------------------
    def _combine_text(self, headers: dict[str, str], body: str) -> str:
        subject = headers.get("Subject") or headers.get("subject") or ""
        subject = subject.strip()
        body = (body or "").strip()
        if subject and body:
            return f"{subject}\n{body}"
        return subject or body

    def score(self, headers: dict[str, str], body: str) -> float:
        model = self._load_model()
        text = self._combine_text(headers, body)
        if not text:
            return 0.0

        tokens = [token for token in _tokenize(text) if token]
        if not tokens:
            return 0.0

        vocab_size = max(1, int(model.get("vocabulary_size") or 0))
        token_counts = model.get("token_counts", {})
        pos_counts: dict[str, int] = token_counts.get("1", {})
        neg_counts: dict[str, int] = token_counts.get("0", {})
        total_tokens = model.get("total_tokens", {})
        total_pos = max(0, int(total_tokens.get("1", 0)))
        total_neg = max(0, int(total_tokens.get("0", 0)))
        class_priors = model.get("class_priors", {})
        prior_pos = float(class_priors.get("1", 1e-9) or 1e-9)
        prior_neg = float(class_priors.get("0", 1e-9) or 1e-9)

        log_pos = math.log(prior_pos)
        log_neg = math.log(prior_neg)
        for token in tokens:
            log_pos += math.log((pos_counts.get(token, 0) + 1) / (total_pos + vocab_size))
            log_neg += math.log((neg_counts.get(token, 0) + 1) / (total_neg + vocab_size))

        max_log = max(log_pos, log_neg)
        pos_prob = math.exp(log_pos - max_log)
        neg_prob = math.exp(log_neg - max_log)
        total = pos_prob + neg_prob
        if total == 0:
            return 0.0
        return pos_prob / total

    def is_relevant(self, headers: dict[str, str], body: str) -> bool:
        score = self.score(headers, body)
        return score >= self.config.threshold

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "LeadMLClassifier":
        return cls(LeadModelConfig.from_env())
