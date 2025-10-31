from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterable


logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[\w']+")

# Stopwords estese IT/EN
EXTENDED_STOPWORDS = {
    # Italiano
    "a", "ai", "al", "alla", "alle", "allo", "anche", "avere", "che", "chi", "ci", "con",
    "cosa", "cui", "da", "dal", "dalla", "dalle", "dallo", "degli", "dei", "del", "della",
    "delle", "dello", "di", "dove", "e", "ed", "essere", "gli", "ha", "hai", "hanno", "ho",
    "i", "il", "in", "io", "la", "le", "lei", "li", "lo", "loro", "lui", "ma", "me", "mi",
    "mia", "mio", "nel", "nella", "nelle", "nello", "noi", "non", "nostro", "o", "per", "però",
    "più", "quale", "quando", "quei", "quelle", "quelli", "quello", "questo", "questi",
    "qui", "se", "sei", "si", "sia", "siamo", "siete", "sono", "sopra", "sotto", "sta",
    "stato", "su", "sua", "sue", "sui", "sul", "sulla", "sulle", "sullo", "suo", "suoi",
    "ti", "tra", "tu", "tua", "tue", "tuo", "tuoi", "tutto", "un", "una", "uno", "va",
    "vai", "voi", "vorrei", "vostro",
    # English
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "can't", "cannot", "could", "couldn't", "did",
    "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few",
    "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
    "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def _tokenize(text: str, stopwords: set[str] = EXTENDED_STOPWORDS) -> list[str]:
    """Tokenizzazione migliorata con stopwords estese."""
    tokens = [token for token in _TOKEN_RE.findall(text.lower()) if token]
    # Rimuovi stopwords e token troppo corti
    return [t for t in tokens if t not in stopwords and len(t) > 2]


def _extract_ngrams(tokens: list[str], n: int = 2) -> list[str]:
    """Estrae n-grammi da lista di token."""
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _extract_features(text: str, headers: dict[str, str]) -> dict[str, float]:
    """Estrae feature numeriche aggiuntive dal testo."""
    text_lower = text.lower()
    
    # Pattern di urgenza
    urgency_words = ['urgente', 'urgent', 'asap', 'immediato', 'subito', 'prima possibile']
    urgency_count = sum(1 for word in urgency_words if word in text_lower)
    
    # Pattern di cortesia
    greeting_words = ['buongiorno', 'buonasera', 'gentile', 'dear', 'hello', 'hi', 'salve']
    has_greeting = any(word in text_lower for word in greeting_words)
    
    # Domande
    question_count = text.count('?')
    
    # Lunghezza
    word_count = len(text.split())
    
    # Pattern di contatto
    has_phone_pattern = bool(re.search(r'\+?\d[\d\s().-]{7,}', text))
    has_email_pattern = bool(re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text))
    
    # Pattern firma (più righe con nome/ruolo/telefono)
    signature_indicators = ['tel', 'phone', 'mobile', 'cell', 'email', '@']
    lines = text.split('\n')
    recent_window = [line for line in lines[-5:] if line.strip()]
    window = recent_window or lines[-5:]
    window_size = max(len(window), 1)
    signature_matches = sum(
        1 for line in window
        if any(ind in line.lower() for ind in signature_indicators)
    )
    signature_score = signature_matches / window_size
    
    return {
        'urgency_score': min(urgency_count / 3.0, 1.0),  # normalizza
        'has_greeting': float(has_greeting),
        'question_density': min(question_count / max(word_count / 50, 1), 1.0),
        'length_score': min(word_count / 200.0, 1.0),  # penalizza troppo brevi/lunghi
        'has_contact_info': float(has_phone_pattern or has_email_pattern),
        'signature_score': signature_score,
    }


class ModelNotAvailableError(RuntimeError):
    """Raised when the ML model cannot be loaded."""


@dataclass
class LeadModelConfig:
    model_path: Path
    threshold: float
    use_ngrams: bool = True
    use_features: bool = True

    @classmethod
    def from_env(cls) -> "LeadModelConfig":
        raw_path = os.getenv("LEAD_MODEL_PATH", "model_store/lead_classifier.json")
        raw_threshold = os.getenv("LEAD_MODEL_THRESHOLD", "0.5")
        use_ngrams = os.getenv("ML_USE_NGRAMS", "true").lower() in ("true", "1", "yes")
        use_features = os.getenv("ML_USE_FEATURES", "true").lower() in ("true", "1", "yes")
        
        try:
            threshold = float(raw_threshold)
        except ValueError:
            threshold = 0.5
        return cls(
            model_path=Path(raw_path), 
            threshold=threshold,
            use_ngrams=use_ngrams,
            use_features=use_features
        )


class LeadMLClassifier:
    """Wrapper around the trained ML model used for lead detection."""

    def __init__(self, config: LeadModelConfig | None = None) -> None:
        self.config = config or LeadModelConfig.from_env()
        self._model: dict | None = None
        self._lock = Lock()

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

    def _combine_text(self, headers: dict[str, str], body: str) -> str:
        subject = headers.get("Subject") or headers.get("subject") or ""
        subject = subject.strip()
        body = (body or "").strip()
        if subject and body:
            return f"{subject}\n{body}"
        return subject or body

    def score(self, headers: dict[str, str], body: str) -> float:
        """Calcola probabilità che sia un lead con feature migliorate."""
        model = self._load_model()
        text = self._combine_text(headers, body)
        if not text:
            return 0.0

        # Tokenizza
        tokens = _tokenize(text, EXTENDED_STOPWORDS)
        if not tokens:
            return 0.0

        # Estrai parametri modello
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

        # Calcola log-probability base (unigrams)
        log_pos = math.log(prior_pos)
        log_neg = math.log(prior_neg)
        
        for token in tokens:
            log_pos += math.log((pos_counts.get(token, 0) + 1) / (total_pos + vocab_size))
            log_neg += math.log((neg_counts.get(token, 0) + 1) / (total_neg + vocab_size))

        # Aggiungi bigrams se abilitato
        if self.config.use_ngrams:
            bigrams = _extract_ngrams(tokens, n=2)
            bigram_counts_pos = model.get("bigram_counts", {}).get("1", {})
            bigram_counts_neg = model.get("bigram_counts", {}).get("0", {})
            
            for bigram in bigrams:
                # Peso ridotto per bigrams (0.5x rispetto a unigrams)
                log_pos += 0.5 * math.log((bigram_counts_pos.get(bigram, 0) + 1) / (total_pos + vocab_size))
                log_neg += 0.5 * math.log((bigram_counts_neg.get(bigram, 0) + 1) / (total_neg + vocab_size))

        # Aggiungi feature numeriche se abilitate
        if self.config.use_features:
            features = _extract_features(text, headers)
            feature_weights = model.get("feature_weights", {})
            
            for feat_name, feat_value in features.items():
                weight = feature_weights.get(feat_name, 0.0)
                # Incremento log-prob in base a feature * weight
                log_pos += feat_value * weight
                # Le feature negative pesano meno
                log_neg -= feat_value * weight * 0.5

        # Converti in probabilità
        max_log = max(log_pos, log_neg)
        pos_prob = math.exp(log_pos - max_log)
        neg_prob = math.exp(log_neg - max_log)
        total = pos_prob + neg_prob
        if total == 0:
            return 0.0
        
        probability = pos_prob / total
        
        # Log confidence per debugging
        if probability > 0.9 or probability < 0.1:
            logger.debug(
                "High confidence prediction",
                extra={
                    "probability": probability,
                    "subject": headers.get("Subject", "")[:50],
                    "token_count": len(tokens)
                }
            )
        
        return probability

    def is_relevant(self, headers: dict[str, str], body: str) -> bool:
        score = self.score(headers, body)
        return score >= self.config.threshold

    def score_with_confidence(self, headers: dict[str, str], body: str) -> tuple[float, str]:
        """Ritorna (score, confidence_level)."""
        score = self.score(headers, body)
        
        if score >= 0.9:
            confidence = "high"
        elif score >= 0.7:
            confidence = "medium"
        elif score >= 0.5:
            confidence = "low"
        else:
            confidence = "very_low"
        
        return score, confidence

    @classmethod
    def from_env(cls) -> "LeadMLClassifier":
        return cls(LeadModelConfig.from_env())