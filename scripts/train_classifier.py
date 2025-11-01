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
    from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
    from sklearn.feature_extraction import DictVectorizer  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed in minimal environments
    BaseEstimator = object  # type: ignore[misc,assignment]
    TransformerMixin = object  # type: ignore[misc,assignment]
    DictVectorizer = None  # type: ignore[assignment]
    TfidfVectorizer = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    Pipeline = None  # type: ignore[assignment]
    precision_recall_curve = roc_curve = roc_auc_score = None  # type: ignore[assignment]

from scipy import sparse


logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[\w']+")

# Stopwords estese (IT + EN)
EXTENDED_STOPWORDS = {
    # Italiano
    "a", "ai", "al", "alla", "alle", "allo", "anche", "avere", "che", "chi", "ci", "con",
    "cosa", "cui", "da", "dal", "dalla", "dalle", "dallo", "degli", "dei", "del", "della",
    "delle", "dello", "di", "dove", "e", "ed", "essere", "gli", "ha", "hai", "hanno", "ho",
    "i", "il", "in", "io", "la", "le", "lei", "li", "lo", "loro", "lui", "ma", "me", "mi",
    "mio", "nel", "nella", "nelle", "nello", "noi", "non", "nostro", "o", "per", "però",
    "più", "quale", "quando", "quei", "quelle", "quelli", "quello", "questo", "questi",
    "qui", "se", "sei", "si", "sia", "siamo", "siete", "sono", "sopra", "sotto", "sta",
    "stato", "su", "sua", "sue", "sui", "sul", "sulla", "sulle", "sullo", "suo", "suoi",
    "ti", "tra", "tu", "tua", "tue", "tuo", "tuoi", "tutto", "un", "una", "uno", "va",
    "vai", "voi", "vostro",
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
    "yourself", "yourselves",
}


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
    text_lower = text.lower()
    urgency_words = ["urgente", "urgent", "asap", "immediato", "subito", "prima possibile"]
    urgency_count = sum(1 for word in urgency_words if word in text_lower)
    greeting_words = ["buongiorno", "buonasera", "gentile", "dear", "hello", "hi", "salve"]
    has_greeting = any(word in text_lower for word in greeting_words)
    question_count = text.count("?")
    word_count = len(text.split())
    has_phone = bool(re.search(r"\+?\d[\d\s().-]{7,}", text))
    has_email = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text))
    lines = [line for line in text.split("\n") if line.strip()]
    window = lines[-5:] if lines else []
    signature_indicators = ["tel", "phone", "mobile", "cell", "email", "@"]
    signature_matches = sum(
        1 for line in window if any(ind in line.lower() for ind in signature_indicators)
    )
    signature_score = signature_matches / max(len(window), 5)

    return {
        "urgency_score": min(urgency_count / 3.0, 1.0),
        "has_greeting": float(has_greeting),
        "question_density": min(question_count / max(word_count / 50, 1), 1.0),
        "length_score": min(word_count / 200.0, 1.0),
        "has_contact_info": float(has_phone or has_email),
        "signature_score": signature_score,
    }


def _feature_tokens_from_text(text: str) -> list[str]:
    feature_tokens: list[str] = []
    for name, value in _extract_features(text).items():
        if value <= 0:
            continue
        repeats = max(1, int(round(value * 3)))
        feature_tokens.extend([f"__feat_{name}"] * repeats)
    return feature_tokens


class _TfidfAnalyzer:
    """Tokenizer compatible with TF-IDF mirroring the NB preprocessing."""

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
                new_subject = re.sub(
                    rf"\b{re.escape(original)}\b", replacement, subject, flags=re.IGNORECASE
                )
                new_body = re.sub(
                    rf"\b{re.escape(original)}\b", replacement, body, flags=re.IGNORECASE
                )
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
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
            if "label" not in record:
                raise ValueError(f"Missing 'label' field at line {line_number}")

            if augment and record.get("label") == 1:
                records.extend(_augment_record(record))
            else:
                records.append(record)

    if not records:
        raise ValueError("Dataset is empty")

    logger.info("Loaded %d records (augmentation: %s)", len(records), augment)
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


def _train_naive_bayes(train_set: list[dict[str, str]], config: TrainingConfig):
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


@dataclass
class _SimpleLogisticPipeline:
    vocabulary: dict[str, int]
    extra_features: list[str]
    weights: list[float]
    bias: float
    use_ngrams: bool
    use_features: bool

    def _vectorize(self, text: str) -> list[float]:
        features = [0.0] * len(self.weights)
        tokens = _tokenize(text)

        vocab_size = len(self.vocabulary)
        if vocab_size:
            for token in tokens:
                idx = self.vocabulary.get(token)
                if idx is not None:
                    features[idx] += 1.0
            if self.use_ngrams:
                for bigram in _extract_ngrams(tokens, n=2):
                    idx = self.vocabulary.get(bigram)
                    if idx is not None:
                        features[idx] += 1.0
            norm = math.sqrt(sum(value * value for value in features[:vocab_size]))
            if norm:
                for i in range(vocab_size):
                    features[i] /= norm

        if self.use_features and self.extra_features:
            numeric = _extract_features(text)
            base = vocab_size
            for offset, name in enumerate(self.extra_features):
                features[base + offset] = float(numeric.get(name, 0.0))

        return features

    def predict_proba(self, texts: Sequence[str]) -> list[list[float]]:  # type: ignore[override]
        results: list[list[float]] = []
        for text in texts:
            vector = self._vectorize(text)
            score = sum(w * x for w, x in zip(self.weights, vector)) + self.bias
            score = max(-50.0, min(50.0, score))
            prob = 1.0 / (1.0 + math.exp(-score))
            results.append([1.0 - prob, prob])
        return results


def _train_simple_logistic(train_set: list[dict[str, str]], config: TrainingConfig) -> _SimpleLogisticPipeline:
    texts = [_prepare_text(record) for record in train_set]
    labels = [float(int(record.get("label", 0))) for record in train_set]

    token_sequences: list[list[str]] = []
    vocabulary: dict[str, int] = {}
    for text in texts:
        tokens = _tokenize(text)
        augmented = list(tokens)
        if config.use_ngrams:
            augmented.extend(_extract_ngrams(tokens, n=2))
        token_sequences.append(augmented)
        for token in augmented:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

    extra_features: list[str] = []
    if config.use_features:
        seen = set()
        for text in texts:
            for name in _extract_features(text).keys():
                if name not in seen:
                    seen.add(name)
                    extra_features.append(name)

    vocab_size = len(vocabulary)
    extra_size = len(extra_features) if config.use_features else 0
    num_features = max(1, vocab_size + extra_size)

    matrix: list[list[float]] = []
    for text, tokens in zip(texts, token_sequences):
        row = [0.0] * num_features
        for token in tokens:
            idx = vocabulary.get(token)
            if idx is not None:
                row[idx] += 1.0
        if vocab_size:
            norm = math.sqrt(sum(value * value for value in row[:vocab_size]))
            if norm:
                for i in range(vocab_size):
                    row[i] /= norm
        if config.use_features and extra_features:
            numeric = _extract_features(text)
            for offset, name in enumerate(extra_features):
                row[vocab_size + offset] = float(numeric.get(name, 0.0))
        matrix.append(row)

    weights = [0.0] * num_features
    bias = 0.0
    rng = random.Random(config.random_state)
    indices = list(range(len(texts)))

    if not indices:
        return _SimpleLogisticPipeline(vocabulary, extra_features, weights, bias, config.use_ngrams, config.use_features)

    for epoch in range(400):
        rng.shuffle(indices)
        grad_w = [0.0] * num_features
        grad_b = 0.0
        for idx in indices:
            row = matrix[idx]
            score = sum(w * x for w, x in zip(weights, row)) + bias
            score = max(-50.0, min(50.0, score))
            pred = 1.0 / (1.0 + math.exp(-score))
            error = pred - labels[idx]
            grad_b += error
            for j, value in enumerate(row):
                grad_w[j] += error * value
        scale = 1.0 / len(indices)
        learning_rate = 0.5 / (1.0 + epoch * 0.05)
        for j in range(num_features):
            grad = grad_w[j] * scale + 0.01 * weights[j]
            weights[j] -= learning_rate * grad
        bias -= learning_rate * grad_b * scale

    return _SimpleLogisticPipeline(
        vocabulary=vocabulary,
        extra_features=extra_features,
        weights=weights,
        bias=bias,
        use_ngrams=config.use_ngrams,
        use_features=config.use_features,
    )


class _CombinedFeaturesTransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Combine TF-IDF text features with optional engineered features."""

    def __init__(self, use_ngrams: bool, use_features: bool) -> None:
        self.use_ngrams = use_ngrams
        self.use_features = use_features
        self._vectorizer: TfidfVectorizer | None = None
        self._dict_vectorizer: DictVectorizer | None = None

    def fit(self, X: Sequence[str], y: Sequence[int] | None = None):  # noqa: D401
        if TfidfVectorizer is None:
            raise RuntimeError("scikit-learn is required for the TF-IDF + Logistic Regression pipeline.")

        texts = list(X)
        analyzer = _TfidfAnalyzer(self.use_features, self.use_ngrams)
        self._vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            min_df=1,
            lowercase=False,
        )
        self._vectorizer.fit(texts)

        if self.use_features:
            if DictVectorizer is None:
                raise RuntimeError(
                    "scikit-learn DictVectorizer is required when use_features is enabled."
                )
            self._dict_vectorizer = DictVectorizer(sparse=True)
            feature_dicts = [_extract_features(text) for text in texts]
            self._dict_vectorizer.fit(feature_dicts)

        return self

    def transform(self, X: Sequence[str]):  # noqa: D401
        if self._vectorizer is None:
            raise RuntimeError("CombinedFeaturesTransformer must be fitted before use.")

        texts = list(X)
        tfidf_matrix = self._vectorizer.transform(texts)

        if not self.use_features or self._dict_vectorizer is None:
            return tfidf_matrix

        feature_dicts = [_extract_features(text) for text in texts]
        feature_matrix = self._dict_vectorizer.transform(feature_dicts)
        return sparse.hstack([tfidf_matrix, feature_matrix], format="csr")


def _train_logistic_regression(train_set: list[dict[str, str]], config: TrainingConfig):
    if Pipeline is None or TfidfVectorizer is None or LogisticRegression is None:
        logger.info("scikit-learn not available, using simplified logistic regression")
        return _train_simple_logistic(train_set, config)

    if config.use_features and DictVectorizer is None:
        raise RuntimeError("scikit-learn DictVectorizer is required when use_features is enabled.")

    texts = [_prepare_text(record) for record in train_set]
    labels = [int(record.get("label", 0)) for record in train_set]

    transformer = _CombinedFeaturesTransformer(
        use_ngrams=config.use_ngrams,
        use_features=config.use_features,
    )

    pipeline = Pipeline(
        steps=[
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
    y_true: Sequence[int], y_scores: Sequence[float], threshold: float = 0.5
) -> tuple[dict[str, float], dict[str, list[float]], dict[str, list[float]]]:
    y_true_list = [int(value) for value in y_true]
    y_scores_list = [float(value) for value in y_scores]
    y_pred = [1 if score >= threshold else 0 for score in y_scores_list]

    tp, fp, tn, fn = _confusion_matrix(y_true_list, y_pred)

    total = len(y_true_list)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    if roc_curve is not None and roc_auc_score is not None:
        fpr, tpr, roc_thresholds = roc_curve(y_true_list, y_scores_list)
        roc_auc = float(roc_auc_score(y_true_list, y_scores_list))
        roc_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        }
    else:
        fpr, tpr, roc_thresholds = _fallback_roc_curve(y_true_list, y_scores_list)
        roc_auc = _safe_auc(fpr, tpr)
        roc_data = {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds}

    if precision_recall_curve is not None:
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true_list, y_scores_list)
        pr_data = {
            "precision": pr_precision.tolist(),
            "recall": pr_recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        }
    else:
        pr_precision, pr_recall, pr_thresholds = _fallback_precision_recall(y_true_list, y_scores_list)
        pr_data = {
            "precision": pr_precision,
            "recall": pr_recall,
            "thresholds": pr_thresholds,
        }

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "roc_auc": roc_auc,
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
    except FileNotFoundError:
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
        "Dataset split: train=%d (pos=%d, neg=%d), test=%d (pos=%d, neg=%d)",
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

        if Pipeline is None or not isinstance(pipeline, Pipeline):
            algorithm_label = "simple_logreg"
        else:
            algorithm_label = "tfidf_logreg"

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


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    cfg = TrainingConfig.from_args(args)
    train(cfg)
