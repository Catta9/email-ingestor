"""Lead classification utilities."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable


DEFAULT_KEYWORDS = [
    "preventivo",
    "quotazione",
    "prezzo",
    "offerta",
    "proposal",
    "estimate",
]

DEFAULT_SYNONYMS = {
    "preventivo": [
        "preventivi",
        "preventiva",
        "richiesta preventivo",
        "richiesta di preventivo",
        "quote request",
        "request a quote",
        "quote richiesti",
        "stima costi",
        "cost estimate",
        "estimate request",
    ],
    "quotazione": [
        "quotazioni",
        "quotation",
        "quoting",
    ],
    "offerta": [
        "offerte",
        "offer",
        "proposal",
        "proposta",
        "proposte",
    ],
    "prezzo": [
        "prezzi",
        "pricing",
        "price",
        "costo",
        "costi",
        "cost",
    ],
    "estimate": [
        "estimating",
        "estimates",
        "estimation",
    ],
}

STOPWORDS = {
    "a",
    "ai",
    "al",
    "alla",
    "alle",
    "and",
    "are",
    "as",
    "at",
    "con",
    "da",
    "dal",
    "dalla",
    "dalle",
    "dello",
    "dei",
    "delle",
    "di",
    "do",
    "does",
    "gli",
    "he",
    "i",
    "il",
    "in",
    "is",
    "it",
    "la",
    "le",
    "li",
    "lo",
    "loro",
    "ma",
    "mi",
    "nel",
    "nella",
    "nelle",
    "o",
    "of",
    "on",
    "per",
    "quei",
    "quelle",
    "questo",
    "sono",
    "su",
    "the",
    "to",
    "una",
    "un",
    "uno",
    "we",
    "you",
}


@dataclass
class SegmentWeights:
    subject: float = 2.0
    body: float = 1.0
    headers: float = 0.5


class LeadRelevanceScorer:
    """Compute a lead relevance score starting from subject/body/header text."""

    _STEM_PATTERNS = (
        ("preventiv", "preventivo"),
        ("quotaz", "quotazione"),
        ("offert", "offerta"),
        ("propost", "proposta"),
        ("prezz", "prezzo"),
        ("cost", "costo"),
        ("estim", "estimate"),
        ("budg", "budget"),
    )

    def __init__(
        self,
        keywords: Iterable[str] | None = None,
        *,
        synonyms: dict[str, Iterable[str]] | None = None,
        negative_keywords: Iterable[str] | None = None,
        weights: SegmentWeights | None = None,
        threshold: float = 2.0,
    ) -> None:
        self.weights = weights or SegmentWeights()
        self.threshold = threshold

        base_keywords = list(keywords or DEFAULT_KEYWORDS)
        self.keywords = {self._normalize_raw(keyword) for keyword in base_keywords}
        self.token_lookup: dict[str, str] = {
            keyword: keyword for keyword in self.keywords
        }

        synonyms = {**DEFAULT_SYNONYMS, **(synonyms or {})}
        self.phrase_lookup: list[tuple[str, str]] = []
        for canonical, words in synonyms.items():
            canonical_norm = self._normalize_raw(canonical)
            self.keywords.add(canonical_norm)
            self.token_lookup.setdefault(canonical_norm, canonical_norm)
            for word in words:
                phrase_norm = self._normalize_phrase(word)
                if " " in phrase_norm:
                    self.phrase_lookup.append((phrase_norm, canonical_norm))
                else:
                    token_norm = self._normalize_raw(phrase_norm)
                    self.token_lookup[token_norm] = canonical_norm

        self.negative_phrases = []
        for item in negative_keywords or ():
            normalized = self._normalize_phrase(item)
            if normalized:
                self.negative_phrases.append(normalized)

    @classmethod
    def from_env(cls) -> "LeadRelevanceScorer":
        raw_keywords = os.getenv("LEAD_KEYWORDS")
        if raw_keywords:
            keywords = [part.strip().lower() for part in raw_keywords.split(",") if part.strip()]
        else:
            keywords = DEFAULT_KEYWORDS

        raw_negative = os.getenv("LEAD_NEGATIVE_KEYWORDS", "")
        negative_keywords = [
            part.strip().lower() for part in raw_negative.split(",") if part.strip()
        ]

        try:
            threshold = float(os.getenv("LEAD_SCORE_THRESHOLD", "2.0"))
        except ValueError:
            threshold = 2.0

        return cls(
            keywords=keywords,
            negative_keywords=negative_keywords,
            threshold=threshold,
        )

    def score(self, headers: dict[str, str], body: str) -> float:
        """Return a weighted relevance score for the provided email content."""
        subject = headers.get("Subject") or headers.get("subject") or ""
        other_headers = " ".join(
            value
            for key, value in headers.items()
            if key.lower() != "subject" and isinstance(value, str)
        )

        if self._has_negative_context(subject, body, other_headers):
            return 0.0

        subject_score = self._segment_score(subject) * self.weights.subject
        body_score = self._segment_score(body) * self.weights.body
        header_score = self._segment_score(other_headers) * self.weights.headers
        return subject_score + body_score + header_score

    # Helpers -----------------------------------------------------------------
    def _normalize_raw(self, token: str) -> str:
        token = token.lower()
        for prefix, replacement in self._STEM_PATTERNS:
            if token.startswith(prefix):
                return replacement
        return token

    def _tokenize(self, text: str) -> list[str]:
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        parts = [part for part in clean.split() if part]
        tokens = []
        for part in parts:
            if part in STOPWORDS:
                continue
            normalized = self._normalize_raw(part)
            canonical = self.token_lookup.get(normalized, normalized)
            tokens.append(canonical)
        return tokens

    def _normalize_phrase(self, text: str) -> str:
        tokens = self._tokenize(text)
        return " ".join(tokens)

    def _segment_score(self, text: str) -> float:
        if not text:
            return 0.0
        tokens = self._tokenize(text)
        score = sum(1 for token in tokens if token in self.keywords)
        normalized_text = " ".join(tokens)
        for phrase, canonical in self.phrase_lookup:
            if phrase and phrase in normalized_text:
                score += 1
        return float(score)

    def _has_negative_context(self, subject: str, body: str, headers: str) -> bool:
        if not self.negative_phrases:
            return False
        combined = " ".join(
            filter(None, [self._normalize_phrase(subject), self._normalize_phrase(body), self._normalize_phrase(headers)])
        )
        return any(phrase in combined for phrase in self.negative_phrases)

    def is_relevant(self, headers: dict[str, str], body: str) -> bool:
        return self.score(headers, body) >= self.threshold
