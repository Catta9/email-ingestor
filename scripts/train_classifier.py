from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from libs.compat import joblib

try:  # pragma: no cover - optional dependency
    from scipy import sparse  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed when SciPy missing
    sparse = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
    from sklearn.feature_extraction import DictVectorizer  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed in minimal environments
    BaseEstimator = TransformerMixin = object  # type: ignore[assignment]
    DictVectorizer = TfidfVectorizer = LogisticRegression = Pipeline = None  # type: ignore[assignment]
    roc_curve = roc_auc_score = precision_recall_curve = None  # type: ignore[assignment]

if roc_curve is None or roc_auc_score is None or precision_recall_curve is None:  # pragma: no cover - fallback
    from libs.simple_metrics import precision_recall_curve as _simple_precision_recall_curve
    from libs.simple_metrics import roc_auc_score as _simple_roc_auc_score
    from libs.simple_metrics import roc_curve as _simple_roc_curve
else:  # pragma: no cover - imported above when sklearn available
    _simple_precision_recall_curve = None
    _simple_roc_auc_score = None
    _simple_roc_curve = None

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[\w']+")

# Stopwords estese (stesso set del classifier)
extended_it = {
    "a",
    "ai",
    "al",
    "alla",
    "alle",
    "allo",
    "anche",
    "avere",
    "che",
    "chi",
    "ci",
    "con",
    "cosa",
    "cui",
    "da",
    "dal",
    "dalla",
    "dalle",
    "dallo",
    "degli",
    "dei",
    "del",
    "della",
    "delle",
    "dello",
    "di",
    "dove",
    "e",
    "ed",
    "essere",
    "gli",
    "ha",
    "hai",
    "hanno",
    "ho",
    "i",
    "il",
    "in",
    "io",
    "la",
    "le",
    "lei",
    "li",
    "lo",
    "loro",
    "lui",
    "ma",
    "me",
    "mi",
    "mio",
    "nel",
    "nella",
    "nelle",
    "nello",
    "noi",
    "non",
    "nostro",
    "o",
    "per",
    "però",
    "più",
    "quale",
    "quando",
    "quei",
    "quelle",
    "quelli",
    "quello",
    "questo",
    "questi",
    "qui",
    "se",
    "sei",
    "si",
    "sia",
    "siamo",
    "siete",
    "sono",
    "sopra",
    "sotto",
    "sta",
    "stato",
    "su",
    "sua",
    "sue",
    "sui",
    "sul",
    "sulla",
    "sulle",
    "sullo",
    "suo",
    "suoi",
    "ti",
    "tra",
    "tu",
    "tua",
    "tue",
    "tuo",
    "tuoi",
    "tutto",
    "un",
    "una",
    "uno",
    "va",
    "vai",
    "voi",
    "vostro",
}
extended_en = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
}
EXTENDED_STOPWORDS = extended_it | extended_en


@dataclass
class TrainingConfig:
    dataset_path: Path
    output_path: Path
    test_size: float = 0.2
    random_state: int = 42
    use_ngrams: bool = True
    use_features: bool = True
    augment_data: bool = False
    algorithm: str = "naive_bayes"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        dataset_path = Path(args.dataset).expanduser().resolve()
        output_dir = Path(args.output).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.algorithm == "logreg":
            output_path = output_dir / "lead_classifier.joblib"
        else:
            output_path = output_dir / "lead_classifier.json"
        return cls(
            dataset_path=dataset_path,
            output_path=output_path,
            test_size=args.test_size,
            random_state=args.random_state,
            use_ngrams=args.use_ngrams,
            use_features=args.use_features,
            augment_data=args.augment,
            algorithm=args.algorithm,
        )


def _tokenize(text: str) -> list[str]:
    tokens = [token for token in _TOKEN_RE.findall(text.lower()) if token]
    return [t for t in tokens if t not in EXTENDED_STOPWORDS and len(t) > 2]


def _extract_ngrams(tokens: list[str], n: int = 2) -> list[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _extract_features(text: str) -> dict[str, float]:
    """Estrae feature numeriche (stesso del classifier)."""
    text_lower = text.lower()
    urgency_words = ["urgente", "urgent", "asap", "immediato", "subito", "prima possibile"]
    urgency_count = sum(1 for word in urgency_words if word in text_lower)
    greeting_words = ["buongiorno", "buonasera", "gentile", "dear", "hello", "hi", "salve"]
    has_greeting = any(word in text_lower for word in greeting_words)
    question_count = text.count("?")
    word_count = len(text.split())
    has_phone = bool(re.search(r"\+?\d[\d\s().-]{7,}", text))
    has_email = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text))
    lines = text.split("\n")
    signature_indicators = ["tel", "phone", "mobile", "cell", "email", "@"]
    signature_score = sum(
        1 for line in lines[-5:] if any(ind in line.lower() for ind in signature_indicators)
    ) / 5.0

    return {
        "urgency_score": min(urgency_count / 3.0, 1.0),
        "has_greeting": float(has_greeting),
        "question_density": min(question_count / max(word_count / 50, 1), 1.0),
        "length_score": min(word_count / 200.0, 1.0),
        "has_contact_info": float(has_phone or has_email),
        "signature_score": signature_score,
    }


def _feature_tokens_from_text(text: str) -> list[str]:
    """Encode continuous features as pseudo tokens for linear models."""
    feature_tokens: list[str] = []
    features = _extract_features(text)
    for name, value in features.items():
        if value <= 0:
            continue
        repeats = max(1, int(round(value * 3)))
        feature_tokens.extend([f"__feat_{name}"] * repeats)
    return feature_tokens


class _TfidfAnalyzer:
    """Tokenizer compatible with TF-IDF that mirrors the NB pre-processing."""

    def __init__(self, use_features: bool, use_ngrams: bool) -> None:
        self.use_features = use_features
        self.use_ngrams = use_ngrams

    def __call__(self, text: str) -> list[str]:
        tokens = _tokenize(text)
        if self.use_ngrams:
            tokens = tokens + _extract_ngrams(tokens, n=2)
        if self.use_features:
            tokens.extend(_feature_tokens_from_text(text))
        return tokens


def _augment_record(record: dict[str, str]) -> list[dict[str, str]]:
    """Data augmentation: genera varianti di un record."""
    augmented = [record]
    synonyms = {
        "preventivo": ["quotazione", "stima", "budget"],
        "richiesta": ["domanda", "richiedo", "vorrei"],
        "urgente": ["immediato", "asap", "subito"],
        "quote": ["quotation", "estimate", "pricing"],
        "request": ["need", "looking for", "inquire"],
    }

    subject = record.get("subject", "")
    body = record.get("body", "")

    for original, replacements in synonyms.items():
        if original in subject.lower() or original in body.lower():
            for replacement in replacements[:1]:
                new_subject = re.sub(r"\b" + re.escape(original) + r"\b", replacement, subject, flags=re.IGNORECASE)
                new_body = re.sub(r"\b" + re.escape(original) + r"\b", replacement, body, flags=re.IGNORECASE)
                if new_subject != subject or new_body != body:
                    augmented.append({
                        "label": record["label"],
                        "subject": new_subject,
                        "body": new_body,
                    })
                    break

    return augmented[:3]


def _load_records(path: Path, augment: bool = False) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:  # pragma: no cover - invalid dataset
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
            if "label" not in record:
                raise ValueError(f"Missing 'label' field at line {line_number}")

            if augment and record.get("label") == 1:
                records.extend(_augment_record(record))
            else:
                records.append(record)

    if not records:
        raise ValueError("Dataset is empty")

    logger.info("Loaded %s records (augmentation: %s)", len(records), augment)
    return records


def _prepare_text(record: dict[str, str]) -> str:
    subject = (record.get("subject") or "").strip()
    body = (record.get("body") or "").strip()
    if subject and body:
        return f"{subject}\n{body}"
    return subject or body


def _split_dataset(
    records: list[dict[str, str]], test_size: float, random_state: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    positives = [rec for rec in records if int(rec.get("label", 0)) == 1]
    negatives = [rec for rec in records if int(rec.get("label", 0)) == 0]
    if not positives or not negatives:
        raise ValueError("Dataset must contain both positive and negative examples")

    rnd = random.Random(random_state)
    rnd.shuffle(positives)
    rnd.shuffle(negatives)

    def _split(group: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        split_point = max(1, int(len(group) * (1 - test_size)))
        return group[:split_point], group[split_point:] or group[-1:]

    pos_train, pos_test = _split(positives)
    neg_train, neg_test = _split(negatives)

    train = pos_train + neg_train
    test = pos_test + neg_test
    rnd.shuffle(train)
    rnd.shuffle(test)
    return train, test


def _count_tokens_and_bigrams(
    records: Iterable[dict[str, str]], use_ngrams: bool
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    counts_pos: dict[str, int] = {}
    counts_neg: dict[str, int] = {}
    bigram_counts_pos: dict[str, int] = {}
    bigram_counts_neg: dict[str, int] = {}

    for record in records:
        tokens = _tokenize(_prepare_text(record))
        target = counts_pos if int(record.get("label", 0)) == 1 else counts_neg

        for token in tokens:
            target[token] = target.get(token, 0) + 1

        if use_ngrams:
            bigrams = _extract_ngrams(tokens, n=2)
            bigram_target = bigram_counts_pos if int(record.get("label", 0)) == 1 else bigram_counts_neg
            for bigram in bigrams:
                bigram_target[bigram] = bigram_target.get(bigram, 0) + 1

    return counts_pos, counts_neg, bigram_counts_pos, bigram_counts_neg


def _compute_feature_weights(
    records: Iterable[dict[str, str]], use_features: bool
) -> dict[str, float]:
    if not use_features:
        return {}

    feature_sums_pos: dict[str, float] = {}
    feature_sums_neg: dict[str, float] = {}
    count_pos = count_neg = 0

    for record in records:
        text = _prepare_text(record)
        features = _extract_features(text)
        is_pos = int(record.get("label", 0)) == 1

        if is_pos:
            count_pos += 1
            for k, v in features.items():
                feature_sums_pos[k] = feature_sums_pos.get(k, 0.0) + v
        else:
            count_neg += 1
            for k, v in features.items():
                feature_sums_neg[k] = feature_sums_neg.get(k, 0.0) + v

    weights: dict[str, float] = {}
    for feat in feature_sums_pos.keys():
        avg_pos = feature_sums_pos[feat] / max(count_pos, 1)
        avg_neg = feature_sums_neg.get(feat, 0.0) / max(count_neg, 1)
        weights[feat] = (avg_pos - avg_neg) * 2.0

    logger.info("Feature weights: %s", weights)
    return weights


def _train_naive_bayes(train_set: list[dict[str, str]], config: TrainingConfig) -> dict:
    pos_counts, neg_counts, bigram_pos, bigram_neg = _count_tokens_and_bigrams(
        train_set, config.use_ngrams
    )
    feature_weights = _compute_feature_weights(train_set, config.use_features)

    total_pos_tokens = sum(pos_counts.values())
    total_neg_tokens = sum(neg_counts.values())
    vocab = set(pos_counts) | set(neg_counts)

    num_pos = sum(1 for record in train_set if int(record.get("label", 0)) == 1)
    num_neg = len(train_set) - num_pos
    total_docs = max(1, len(train_set))

    model: dict[str, object] = {
        "version": 2,
        "class_priors": {
            "1": num_pos / total_docs,
            "0": num_neg / total_docs,
        },
        "token_counts": {
            "1": pos_counts,
            "0": neg_counts,
        },
        "total_tokens": {
            "1": total_pos_tokens,
            "0": total_neg_tokens,
        },
        "vocabulary_size": len(vocab),
    }

    if config.use_ngrams:
        model["bigram_counts"] = {
            "1": bigram_pos,
            "0": bigram_neg,
        }

    if config.use_features:
        model["feature_weights"] = feature_weights

    return model


class _NaivePipelineAdapter:
    def __init__(self, model: dict, use_ngrams: bool, use_features: bool) -> None:
        self._model = model
        self._use_ngrams = use_ngrams
        self._use_features = use_features

    def predict_proba(self, texts: Sequence[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            proba = _predict_proba(self._model, text, self._use_ngrams, self._use_features)
            results.append([1.0 - proba, proba])
        return results


class _SimpleLogisticPipeline:
    def __init__(
        self,
        vocabulary: dict[str, int],
        extra_features: dict[str, int],
        weights: list[float],
        bias: float,
        use_ngrams: bool,
        use_features: bool,
    ) -> None:
        self.vocabulary = vocabulary
        self.extra_features = extra_features
        self.weights = weights
        self.bias = bias
        self.use_ngrams = use_ngrams
        self.use_features = use_features

    def _vectorize(self, text: str) -> list[float]:
        features = [0.0] * len(self.weights)
        tokens = _tokenize(text)
        if self.use_ngrams:
            tokens += _extract_ngrams(tokens, n=2)

        for token in tokens:
            idx = self.vocabulary.get(token)
            if idx is not None:
                features[idx] += 1.0

        if self.use_features:
            numeric = _extract_features(text)
            for name, value in numeric.items():
                idx = self.extra_features.get(name)
                if idx is not None:
                    features[idx] += value

        return features

    def predict_proba(self, texts: Sequence[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            vector = self._vectorize(text)
            score = sum(w * x for w, x in zip(self.weights, vector)) + self.bias
            score = max(-50.0, min(50.0, score))
            proba = 1.0 / (1.0 + math.exp(-score))
            results.append([1.0 - proba, proba])
        return results


def _train_simple_logistic(train_set: list[dict[str, str]], config: TrainingConfig) -> _SimpleLogisticPipeline:
    texts = [_prepare_text(record) for record in train_set]
    labels = [int(record.get("label", 0)) for record in train_set]

    vocabulary: dict[str, int] = {}
    extra_features: dict[str, int] = {}

    for text in texts:
        tokens = _tokenize(text)
        if config.use_ngrams:
            tokens += _extract_ngrams(tokens, n=2)
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
        if config.use_features:
            for name in _extract_features(text).keys():
                if name not in extra_features:
                    extra_features[name] = len(vocabulary) + len(extra_features)

    num_features = len(vocabulary) + len(extra_features)
    if num_features == 0:
        raise ValueError("Dataset does not contain any features to train on")

    matrix: list[list[float]] = []
    for text in texts:
        vector = [0.0] * num_features
        tokens = _tokenize(text)
        if config.use_ngrams:
            tokens += _extract_ngrams(tokens, n=2)
        for token in tokens:
            idx = vocabulary.get(token)
            if idx is not None:
                vector[idx] += 1.0
        if config.use_features:
            for name, value in _extract_features(text).items():
                idx = extra_features.get(name)
                if idx is not None:
                    vector[idx] += value
        matrix.append(vector)

    rng = random.Random(config.random_state)
    weights = [0.0] * num_features
    bias = 0.0
    indices = list(range(len(matrix)))

    for epoch in range(200):
        rng.shuffle(indices)
        learning_rate = 0.5 / (1.0 + epoch * 0.05)
        for idx in indices:
            row = matrix[idx]
            score = sum(w * x for w, x in zip(weights, row)) + bias
            score = max(-50.0, min(50.0, score))
            pred = 1.0 / (1.0 + math.exp(-score))
            error = pred - labels[idx]
            for j, value in enumerate(row):
                if value:
                    weights[j] -= learning_rate * (error * value + 0.01 * weights[j])
            bias -= learning_rate * error

    return _SimpleLogisticPipeline(
        vocabulary=vocabulary,
        extra_features=extra_features,
        weights=weights,
        bias=bias,
        use_ngrams=config.use_ngrams,
        use_features=config.use_features,
    )


class _CombinedFeaturesTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, use_ngrams: bool, use_features: bool) -> None:
        self.use_ngrams = use_ngrams
        self.use_features = use_features
        self._vectorizer: TfidfVectorizer | None = None
        self._dict_vectorizer: DictVectorizer | None = None

    def fit(self, X: Sequence[str], y: Sequence[int] | None = None):  # noqa: D401
        if TfidfVectorizer is None:
            raise RuntimeError(
                "scikit-learn is required for the TF-IDF + Logistic Regression pipeline."
            )

        texts = list(X)
        ngram_range = (1, 2) if self.use_ngrams else (1, 1)
        analyzer = _TfidfAnalyzer(use_features=False, use_ngrams=self.use_ngrams)
        self._vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=1,
            stop_words=list(EXTENDED_STOPWORDS),
            strip_accents="unicode",
            analyzer=analyzer,
        )
        self._vectorizer.fit(texts)

        if self.use_features:
            if DictVectorizer is None:
                raise RuntimeError("scikit-learn is required for feature vectorization.")
            feature_dicts = [_extract_features(text) for text in texts]
            self._dict_vectorizer = DictVectorizer()
            self._dict_vectorizer.fit(feature_dicts)

        return self

    def transform(self, X: Sequence[str]):
        if self._vectorizer is None:
            raise RuntimeError("Transformer not fitted")
        text_matrix = self._vectorizer.transform(X)

        if not self.use_features:
            return text_matrix

        if self._dict_vectorizer is None:
            raise RuntimeError("Transformer not fitted")
        if sparse is None:
            raise RuntimeError("scipy is required to combine sparse feature matrices")

        feature_dicts = [_extract_features(text) for text in X]
        feature_matrix = self._dict_vectorizer.transform(feature_dicts)
        return sparse.hstack([text_matrix, feature_matrix])


def _train_logistic_regression(train_set: list[dict[str, str]], config: TrainingConfig):
    if Pipeline is None or TfidfVectorizer is None or LogisticRegression is None:
        logger.info("scikit-learn not available, using simplified logistic regression")
        return _train_simple_logistic(train_set, config)

    texts = [_prepare_text(record) for record in train_set]
    labels = [int(record.get("label", 0)) for record in train_set]

    transformer = _CombinedFeaturesTransformer(
        use_ngrams=config.use_ngrams,
        use_features=config.use_features,
    )

    pipeline = Pipeline(
        [
            ("features", transformer),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    pipeline.fit(texts, labels)
    return pipeline


def _predict_proba(model: dict, text: str, use_ngrams: bool, use_features: bool) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0

    vocab_size = max(1, int(model.get("vocabulary_size") or 0))
    pos_counts: dict[str, int] = model["token_counts"]["1"]
    neg_counts: dict[str, int] = model["token_counts"]["0"]
    total_pos = max(0, int(model["total_tokens"]["1"]))
    total_neg = max(0, int(model["total_tokens"]["0"]))
    prior_pos = float(model["class_priors"]["1"] or 1e-9)
    prior_neg = float(model["class_priors"]["0"] or 1e-9)

    log_pos = math.log(prior_pos)
    log_neg = math.log(prior_neg)

    for token in tokens:
        pos_count = pos_counts.get(token, 0)
        neg_count = neg_counts.get(token, 0)
        log_pos += math.log((pos_count + 1) / (total_pos + vocab_size))
        log_neg += math.log((neg_count + 1) / (total_neg + vocab_size))

    if use_ngrams and "bigram_counts" in model:
        bigrams = _extract_ngrams(tokens, n=2)
        bigram_pos = model["bigram_counts"]["1"]
        bigram_neg = model["bigram_counts"]["0"]
        for bigram in bigrams:
            log_pos += 0.5 * math.log((bigram_pos.get(bigram, 0) + 1) / (total_pos + vocab_size))
            log_neg += 0.5 * math.log((bigram_neg.get(bigram, 0) + 1) / (total_neg + vocab_size))

    if use_features and "feature_weights" in model:
        features = _extract_features(text)
        weights = model["feature_weights"]
        for feat_name, feat_value in features.items():
            weight = weights.get(feat_name, 0.0)
            log_pos += feat_value * weight
            log_neg -= feat_value * weight * 0.5

    max_log = max(log_pos, log_neg)
    pos_prob = math.exp(log_pos - max_log)
    neg_prob = math.exp(log_neg - max_log)
    total = pos_prob + neg_prob
    if total == 0:
        return 0.0
    return pos_prob / total


def _confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 0 and pred == 0:
            tn += 1
        elif truth == 1 and pred == 0:
            fn += 1
    return tp, fp, tn, fn


def _safe_auc(fpr: Sequence[float], tpr: Sequence[float]) -> float:
    if len(fpr) < 2:
        return 0.0
    area = 0.0
    for idx in range(1, len(fpr)):
        width = fpr[idx] - fpr[idx - 1]
        height = (tpr[idx] + tpr[idx - 1]) / 2
        area += width * height
    return max(0.0, min(1.0, area))


def _fallback_roc_curve(
    y_true: Sequence[int], y_scores: Sequence[float]
) -> tuple[list[float], list[float], list[float]]:
    pairs = sorted(zip(y_scores, y_true), key=lambda item: item[0], reverse=True)
    thresholds: list[float] = []
    tpr_values: list[float] = []
    fpr_values: list[float] = []

    positives = sum(1 for value in y_true if value == 1)
    negatives = len(y_true) - positives
    positives = max(positives, 1)
    negatives = max(negatives, 1)

    for idx, (score, _) in enumerate(pairs):
        threshold = score
        preds = [1 if value >= threshold else 0 for value, _ in pairs]
        tp, fp, tn, fn = _confusion_matrix([label for _, label in pairs], preds)
        tpr_values.append(tp / positives)
        fpr_values.append(fp / negatives)
        thresholds.append(threshold)

    if thresholds:
        thresholds.append(min(thresholds) - 1e-6)
    else:
        thresholds.append(0.0)
    tpr_values.append(0.0)
    fpr_values.append(0.0)

    combined = sorted(zip(fpr_values, tpr_values, thresholds))
    fpr_sorted = [item[0] for item in combined]
    tpr_sorted = [item[1] for item in combined]
    thresholds_sorted = [item[2] for item in combined]
    return fpr_sorted, tpr_sorted, thresholds_sorted


def _fallback_precision_recall(
    y_true: Sequence[int], y_scores: Sequence[float]
) -> tuple[list[float], list[float], list[float]]:
    pairs = sorted(zip(y_scores, y_true), key=lambda item: item[0], reverse=True)
    precisions: list[float] = []
    recalls: list[float] = []
    thresholds: list[float] = []

    positives = sum(1 for value in y_true if value == 1)
    positives = max(positives, 1)

    tp = fp = 0
    last_score = None
    for score, truth in pairs:
        if truth == 1:
            tp += 1
        else:
            fp += 1
        if score != last_score:
            precision = tp / (tp + fp) if (tp + fp) else 1.0
            recall = tp / positives
            precisions.append(precision)
            recalls.append(recall)
            thresholds.append(score)
            last_score = score

    precisions.append(0.0)
    recalls.append(0.0)
    thresholds.append((thresholds[-1] if thresholds else 0.0) - 1e-6)

    return precisions, recalls, thresholds


def _compute_metrics_from_scores(
    y_true: Sequence[int], y_scores: Sequence[float]
) -> tuple[dict[str, float], dict[str, list[float]], dict[str, list[float]]]:
    if len(y_true) != len(y_scores):
        raise ValueError("Predictions and targets must have the same length")

    y_pred = [1 if score >= 0.5 else 0 for score in y_scores]
    tp, fp, tn, fn = _confusion_matrix(y_true, y_pred)

    total = max(len(y_true), 1)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    if roc_curve is not None and roc_auc_score is not None:
        try:
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        except Exception:  # pragma: no cover - fallback when sklearn errors
            fpr, tpr, roc_thresholds = _fallback_roc_curve(y_true, y_scores)
        try:
            roc_auc = float(roc_auc_score(y_true, y_scores))
        except Exception:  # pragma: no cover - fallback when sklearn errors
            roc_auc = _safe_auc(fpr, tpr)
    else:
        fpr, tpr, roc_thresholds = (
            _simple_roc_curve(y_true, y_scores)  # type: ignore[misc]
            if _simple_roc_curve is not None
            else _fallback_roc_curve(y_true, y_scores)
        )
        roc_auc = (
            float(_simple_roc_auc_score(y_true, y_scores))  # type: ignore[misc]
            if _simple_roc_auc_score is not None
            else _safe_auc(fpr, tpr)
        )

    if precision_recall_curve is not None:
        try:
            precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_scores)
        except Exception:  # pragma: no cover - fallback when sklearn errors
            precisions, recalls, pr_thresholds = _fallback_precision_recall(y_true, y_scores)
    else:
        precisions, recalls, pr_thresholds = (
            _simple_precision_recall_curve(y_true, y_scores)  # type: ignore[misc]
            if _simple_precision_recall_curve is not None
            else _fallback_precision_recall(y_true, y_scores)
        )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    roc_data = {
        "fpr": list(fpr),
        "tpr": list(tpr),
        "thresholds": list(roc_thresholds),
    }
    pr_data = {
        "precision": list(precisions),
        "recall": list(recalls),
        "thresholds": list(pr_thresholds),
    }
    return metrics, roc_data, pr_data


def _evaluate(
    model: dict, records: Iterable[dict[str, str]], use_ngrams: bool, use_features: bool
) -> tuple[dict[str, float], dict[str, list[float]], dict[str, list[float]]]:
    scores: list[float] = []
    targets: list[int] = []
    for record in records:
        text = _prepare_text(record)
        proba = _predict_proba(model, text, use_ngrams, use_features)
        scores.append(proba)
        targets.append(int(record.get("label", 0)))
    return _compute_metrics_from_scores(targets, scores)


def _format_metrics(metrics: dict[str, float]) -> str:
    return (
        "Accuracy: {accuracy:.3f}\n"
        "Precision: {precision:.3f}\n"
        "Recall: {recall:.3f}\n"
        "F1-score: {f1:.3f}\n"
        "ROC-AUC: {roc_auc:.3f}\n"
        "TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}\n"
    ).format(**metrics)


def _save_metrics_artifacts(
    output_path: Path,
    metrics: dict[str, float],
    roc_data: dict[str, list[float]],
    pr_data: dict[str, list[float]],
) -> None:
    metrics_text_path = output_path.with_suffix(".metrics.txt")
    metrics_json_path = output_path.with_suffix(".metrics.json")
    curves_json_path = output_path.with_suffix(".curves.json")

    metrics_text_path.write_text(_format_metrics(metrics), encoding="utf-8")

    metrics_payload = {"metrics": metrics}
    metrics_json_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    curves_payload = {"roc": roc_data, "precision_recall": pr_data}
    curves_json_path.write_text(
        json.dumps(curves_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _copy_model_to_default_store(model_path: Path) -> None:
    default_store_env = os.getenv("LEAD_MODEL_PATH")
    if default_store_env:
        default_store = Path(default_store_env)
    else:
        default_store = Path("model_store") / model_path.name

    if default_store.suffix != model_path.suffix:
        default_store = default_store.with_suffix(model_path.suffix)

    try:
        if default_store.resolve() == model_path.resolve():
            return
    except FileNotFoundError:  # pragma: no cover - missing path during resolve
        pass

    default_store.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, default_store)

    metadata_path = model_path.with_suffix(".meta.json")
    if metadata_path.exists():
        shutil.copy2(metadata_path, default_store.with_suffix(".meta.json"))

    logger.info("Copied model to default store: %s", default_store)


def train(config: TrainingConfig) -> None:
    records = _load_records(config.dataset_path, augment=config.augment_data)
    train_set, test_set = _split_dataset(records, config.test_size, config.random_state)

    pos_train = sum(1 for r in train_set if int(r.get("label", 0)) == 1)
    neg_train = len(train_set) - pos_train
    pos_test = sum(1 for r in test_set if int(r.get("label", 0)) == 1)
    neg_test = len(test_set) - pos_test

    logger.info(
        "Dataset split: train=%s (pos=%s, neg=%s), test=%s (pos=%s, neg=%s)",
        len(train_set),
        pos_train,
        neg_train,
        len(test_set),
        pos_test,
        neg_test,
    )

    logger.info("Selected training algorithm: %s", config.algorithm)

    if config.algorithm == "logreg":
        pipeline = _train_logistic_regression(train_set, config)
        model_output_path = config.output_path
        joblib.dump(pipeline, model_output_path)
        logger.info("Persisted TF-IDF + Logistic Regression pipeline to %s", model_output_path)

        texts_test = [_prepare_text(record) for record in test_set]
        y_true = [int(record.get("label", 0)) for record in test_set]
        probabilities = pipeline.predict_proba(texts_test)
        y_scores = [row[1] for row in probabilities]

        metrics, roc_data, pr_data = _compute_metrics_from_scores(y_true, y_scores)
        logger.info("Evaluation metrics:\n%s", _format_metrics(metrics).strip())

        algorithm_label = "tfidf_logreg" if isinstance(pipeline, Pipeline) else "simple_logreg"
        metadata = {
            "algorithm": algorithm_label,
            "version": 1,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "threshold": 0.5,
        }
        metadata_path = model_output_path.with_suffix(".meta.json")
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Saved model metadata to %s", metadata_path)

    elif config.algorithm == "naive_bayes":
        model = _train_naive_bayes(train_set, config)
        metrics, roc_data, pr_data = _evaluate(
            model, test_set, config.use_ngrams, config.use_features
        )
        logger.info("Evaluation metrics:\n%s", _format_metrics(metrics).strip())

        model_output_path = config.output_path
        model_output_path.write_text(
            json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Model persisted to %s", model_output_path)

        metadata = {
            "algorithm": "naive_bayes",
            "version": model.get("version", 1),
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "threshold": 0.5,
        }
        metadata_path = model_output_path.with_suffix(".meta.json")
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")

    if metrics["accuracy"] > 0.95:
        logger.warning(
            "⚠️  Very high accuracy (%.3f) may indicate overfitting. "
            "Consider expanding the dataset or reviewing test/train split.",
            metrics["accuracy"],
        )

    _save_metrics_artifacts(model_output_path, metrics, roc_data, pr_data)
    logger.info("Saved evaluation metrics and curves next to %s", model_output_path)

    _copy_model_to_default_store(model_output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lead classification model")
    parser.add_argument(
        "--dataset",
        default="datasets/lead_training.jsonl",
        help="Path to the JSONL dataset containing email examples",
    )
    parser.add_argument(
        "--output",
        default="artifacts",
        help="Directory where the trained model should be stored",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset reserved for evaluation",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split",
    )
    parser.add_argument(
        "--no-ngrams",
        action="store_true",
        help="Disable bigram features",
    )
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="Disable numerical features",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation (synonym replacement)",
    )
    parser.add_argument(
        "--algorithm",
        choices=["naive_bayes", "logreg"],
        default="naive_bayes",
        help="Algorithm to train: 'naive_bayes' (legacy) or 'logreg' (TF-IDF + Logistic Regression)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. INFO, DEBUG)",
    )
    args = parser.parse_args()
    args.use_ngrams = not args.no_ngrams
    args.use_features = not args.no_features
    return args


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    cfg = TrainingConfig.from_args(args)
    train(cfg)
