"""Ensemble classifier che combina rule-based e ML in modo intelligente."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from .lead_classifier import LeadRelevanceScorer
from .ml_classifier import LeadMLClassifier, ModelNotAvailableError


logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    strategy: str = "hybrid"  # hybrid | ml | rule_based
    ml_threshold: float = 0.5
    rule_threshold: float = 2.0
    ml_weight: float = 0.7  # peso ML quando entrambi disponibili
    confidence_threshold_high: float = 0.9  # sopra: usa solo ML
    confidence_threshold_low: float = 0.3   # sotto: usa solo ML
    
    @classmethod
    def from_env(cls) -> "EnsembleConfig":
        return cls(
            strategy=os.getenv("LEAD_CLASSIFIER_STRATEGY", "hybrid").lower(),
            ml_threshold=float(os.getenv("LEAD_MODEL_THRESHOLD", "0.5")),
            rule_threshold=float(os.getenv("LEAD_SCORE_THRESHOLD", "2.0")),
            ml_weight=float(os.getenv("ENSEMBLE_ML_WEIGHT", "0.7")),
            confidence_threshold_high=float(os.getenv("ENSEMBLE_CONF_HIGH", "0.9")),
            confidence_threshold_low=float(os.getenv("ENSEMBLE_CONF_LOW", "0.3")),
        )


class EnsembleLeadClassifier:
    """
    Combina rule-based e ML con logica intelligente:
    
    1. Se ML ha alta confidenza (>0.9 o <0.1) → usa solo ML
    2. Se ML ha confidenza media → weighted average
    3. Se ML non disponibile → fallback a rule-based
    """
    
    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or EnsembleConfig.from_env()
        self.rule_scorer = LeadRelevanceScorer.from_env()
        
        try:
            self.ml_classifier = LeadMLClassifier.from_env()
            self.ml_available = True
        except (ModelNotAvailableError, FileNotFoundError) as exc:
            logger.warning("ML classifier unavailable, using rule-based only: %s", exc)
            self.ml_classifier = None
            self.ml_available = False
    
    def score(self, headers: dict[str, str], body: str) -> tuple[float, str, dict]:
        """
        Calcola score ensemble.
        
        Returns:
            (score, confidence_level, debug_info)
        """
        debug_info = {"strategy": self.config.strategy}
        
        # Strategy: rule_based only
        if self.config.strategy == "rule_based" or not self.ml_available:
            rule_score = self.rule_scorer.score(headers, body)
            normalized_score = min(rule_score / self.config.rule_threshold, 1.0)
            debug_info.update({
                "rule_score": rule_score,
                "normalized_score": normalized_score,
                "method": "rule_based"
            })
            confidence = "medium" if normalized_score > 0.5 else "low"
            return normalized_score, confidence, debug_info
        
        # Strategy: ML only
        if self.config.strategy == "ml":
            ml_score, ml_confidence = self.ml_classifier.score_with_confidence(headers, body)
            debug_info.update({
                "ml_score": ml_score,
                "ml_confidence": ml_confidence,
                "method": "ml_only"
            })
            return ml_score, ml_confidence, debug_info
        
        # Strategy: Hybrid (default)
        ml_score, ml_confidence = self.ml_classifier.score_with_confidence(headers, body)
        rule_score = self.rule_scorer.score(headers, body)
        
        debug_info.update({
            "ml_score": ml_score,
            "ml_confidence": ml_confidence,
            "rule_score": rule_score,
        })
        
        # Alta confidenza ML → usa solo ML
        if ml_score >= self.config.confidence_threshold_high:
            debug_info["method"] = "ml_high_confidence"
            return ml_score, "high", debug_info
        
        if ml_score <= self.config.confidence_threshold_low:
            debug_info["method"] = "ml_low_confidence"
            return ml_score, "high", debug_info
        
        # Confidenza media → weighted average
        rule_normalized = min(rule_score / self.config.rule_threshold, 1.0)
        ensemble_score = (
            self.config.ml_weight * ml_score + 
            (1 - self.config.ml_weight) * rule_normalized
        )
        
        debug_info.update({
            "rule_normalized": rule_normalized,
            "ensemble_score": ensemble_score,
            "method": "weighted_ensemble"
        })
        
        # Confidence basato su accordo tra metodi
        agreement = abs(ml_score - rule_normalized)
        if agreement < 0.2:
            confidence = "high"
        elif agreement < 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        debug_info["agreement"] = agreement
        
        return ensemble_score, confidence, debug_info
    
    def is_relevant(self, headers: dict[str, str], body: str) -> bool:
        """Determina se è un lead usando ensemble logic."""
        score, confidence, debug = self.score(headers, body)
        
        # Threshold dinamico basato su confidenza
        if confidence == "high":
            threshold = self.config.ml_threshold
        elif confidence == "medium":
            threshold = self.config.ml_threshold + 0.1  # più conservativo
        else:
            threshold = self.config.ml_threshold + 0.15  # molto conservativo
        
        is_lead = score >= threshold
        
        if is_lead and confidence == "low":
            logger.info(
                "Low confidence lead detected",
                extra={
                    "score": score,
                    "confidence": confidence,
                    "threshold": threshold,
                    "debug": debug,
                }
            )
        
        return is_lead
    
    def classify_with_explanation(
        self, headers: dict[str, str], body: str
    ) -> dict[str, any]:
        """Classificazione con spiegazione dettagliata."""
        score, confidence, debug = self.score(headers, body)
        is_lead = self.is_relevant(headers, body)
        
        return {
            "is_lead": is_lead,
            "score": score,
            "confidence": confidence,
            "explanation": debug,
            "threshold": self.config.ml_threshold,
        }