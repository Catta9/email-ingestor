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

import joblib
import numpy as np


try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
    from sklearn.pipeline import Pipeline
except ModuleNotFoundError:  # pragma: no cover - optional dependency for legacy NB only
    TfidfVectorizer = LogisticRegression = Pipeline = None  # type: ignore
    precision_recall_curve = roc_curve = roc_auc_score = None  # type: ignore


logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[\w']+")

# Stopwords estese (stesso set del classifier)
EXTENDED_STOPWORDS = {
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
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _extract_features(text: str) -> dict[str, float]:
    """Estrae feature numeriche (stesso del classifier)."""
    text_lower = text.lower()
    urgency_words = ['urgente', 'urgent', 'asap', 'immediato', 'subito', 'prima possibile']
    urgency_count = sum(1 for word in urgency_words if word in text_lower)
    greeting_words = ['buongiorno', 'buonasera', 'gentile', 'dear', 'hello', 'hi', 'salve']
    has_greeting = any(word in text_lower for word in greeting_words)
    question_count = text.count('?')
    word_count = len(text.split())
    has_phone = bool(re.search(r'\+?\d[\d\s().-]{7,}', text))
    has_email = bool(re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text))
    lines = text.split('\n')
    signature_indicators = ['tel', 'phone', 'mobile', 'cell', 'email', '@']
    signature_score = sum(
        1 for line in lines[-5:] if any(ind in line.lower() for ind in signature_indicators)
    ) / 5.0
    
    return {
        'urgency_score': min(urgency_count / 3.0, 1.0),
        'has_greeting': float(has_greeting),
        'question_density': min(question_count / max(word_count / 50, 1), 1.0),
        'length_score': min(word_count / 200.0, 1.0),
        'has_contact_info': float(has_phone or has_email),
        'signature_score': signature_score,
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
    augmented = [record]  # include originale
    
    # Sinonimi comuni IT/EN
    synonyms = {
        'preventivo': ['quotazione', 'stima', 'budget'],
        'richiesta': ['domanda', 'richiedo', 'vorrei'],
        'urgente': ['immediato', 'asap', 'subito'],
        'quote': ['quotation', 'estimate', 'pricing'],
        'request': ['need', 'looking for', 'inquire'],
    }
    
    subject = record.get('subject', '')
    body = record.get('body', '')
    
    # Genera 1-2 varianti con sinonimi
    for original, replacements in synonyms.items():
        if original in subject.lower() or original in body.lower():
            for replacement in replacements[:1]:  # usa solo 1 sinonimo
                new_subject = re.sub(
                    r'\b' + re.escape(original) + r'\b', 
                    replacement, 
                    subject, 
                    flags=re.IGNORECASE
                )
                new_body = re.sub(
                    r'\b' + re.escape(original) + r'\b', 
                    replacement, 
                    body, 
                    flags=re.IGNORECASE
                )
                if new_subject != subject or new_body != body:
                    augmented.append({
                        'label': record['label'],
                        'subject': new_subject,
                        'body': new_body
                    })
                    break  # 1 variante per sinonimo
    
    return augmented[:3]  # max 3 varianti (originale + 2)


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
            
            if augment and record.get('label') == 1:  # solo lead positivi
                records.extend(_augment_record(record))
            else:
                records.append(record)
    
    if not records:
        raise ValueError("Dataset is empty")
    
    logger.info(f"Loaded {len(records)} records (augmentation: {augment})")
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
    """Conta unigrams e bigrams per classe."""
    counts_pos: dict[str, int] = {}
    counts_neg: dict[str, int] = {}
    bigram_counts_pos: dict[str, int] = {}
    bigram_counts_neg: dict[str, int] = {}
    
    for record in records:
        tokens = _tokenize(_prepare_text(record))
        target = counts_pos if int(record.get("label", 0)) == 1 else counts_neg
        
        # Unigrams
        for token in tokens:
            target[token] = target.get(token, 0) + 1
        
        # Bigrams
        if use_ngrams:
            bigrams = _extract_ngrams(tokens, n=2)
            bigram_target = bigram_counts_pos if int(record.get("label", 0)) == 1 else bigram_counts_neg
            for bigram in bigrams:
                bigram_target[bigram] = bigram_target.get(bigram, 0) + 1
    
    return counts_pos, counts_neg, bigram_counts_pos, bigram_counts_neg


def _compute_feature_weights(
    records: Iterable[dict[str, str]], use_features: bool
) -> dict[str, float]:
    """Calcola correlazione feature -> lead per pesatura."""
    if not use_features:
        return {}
    
    feature_sums_pos = {}
    feature_sums_neg = {}
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
    
    # Calcola media per classe
    weights = {}
    for feat in feature_sums_pos.keys():
        avg_pos = feature_sums_pos[feat] / max(count_pos, 1)
        avg_neg = feature_sums_neg.get(feat, 0) / max(count_neg, 1)
        # Peso = differenza normalizzata
        weights[feat] = (avg_pos - avg_neg) * 2.0  # scala per effetto maggiore
    
    logger.info(f"Feature weights: {weights}")
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

    model = {
        "version": 2,  # versione aggiornata
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


def _train_logistic_regression(
    train_set: list[dict[str, str]], config: TrainingConfig
) -> Pipeline:
    if Pipeline is None or TfidfVectorizer is None or LogisticRegression is None:
        raise RuntimeError(
            "scikit-learn is required for the TF-IDF + Logistic Regression pipeline."
        )

    texts = [_prepare_text(record) for record in train_set]
    labels = [int(record.get("label", 0)) for record in train_set]

    analyzer = _TfidfAnalyzer(
        use_features=config.use_features, use_ngrams=config.use_ngrams
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=analyzer,
                    min_df=1,
                    lowercase=False,
                ),
            ),
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
    """Predizione con bigrams e features."""
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

    # Unigrams
    for token in tokens:
        pos_count = pos_counts.get(token, 0)
        neg_count = neg_counts.get(token, 0)
        log_pos += math.log((pos_count + 1) / (total_pos + vocab_size))
        log_neg += math.log((neg_count + 1) / (total_neg + vocab_size))

    # Bigrams
    if use_ngrams and "bigram_counts" in model:
        bigrams = _extract_ngrams(tokens, n=2)
        bigram_pos = model["bigram_counts"]["1"]
        bigram_neg = model["bigram_counts"]["0"]
        for bigram in bigrams:
            log_pos += 0.5 * math.log((bigram_pos.get(bigram, 0) + 1) / (total_pos + vocab_size))
            log_neg += 0.5 * math.log((bigram_neg.get(bigram, 0) + 1) / (total_neg + vocab_size))

    # Features
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


def _compute_metrics_from_scores(
    y_true: Sequence[int], y_scores: Sequence[float], threshold: float = 0.5
) -> tuple[dict[str, float], dict[str, list[float]], dict[str, list[float]]]:
    if precision_recall_curve is None or roc_curve is None or roc_auc_score is None:
        raise RuntimeError(
            "scikit-learn is required to compute evaluation metrics. Install scikit-learn first."
        )

    y_true_array = np.asarray(y_true)
    y_scores_array = np.asarray(y_scores)
    y_pred = (y_scores_array >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true_array == 1)).sum())
    fp = int(((y_pred == 1) & (y_true_array == 0)).sum())
    tn = int(((y_pred == 0) & (y_true_array == 0)).sum())
    fn = int(((y_pred == 0) & (y_true_array == 1)).sum())

    total = len(y_true_array)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    fpr, tpr, roc_thresholds = roc_curve(y_true_array, y_scores_array)
    roc_auc = roc_auc_score(y_true_array, y_scores_array)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true_array, y_scores_array)

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
    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": roc_thresholds.tolist(),
    }
    pr_data = {
        "precision": pr_precision.tolist(),
        "recall": pr_recall.tolist(),
        "thresholds": pr_thresholds.tolist(),
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

    pos_train = sum(1 for r in train_set if r.get('label') == 1)
    neg_train = len(train_set) - pos_train
    pos_test = sum(1 for r in test_set if r.get('label') == 1)
    neg_test = len(test_set) - pos_test
    
    logger.info(
        f"Dataset split: train={len(train_set)} (pos={pos_train}, neg={neg_train}), "
        f"test={len(test_set)} (pos={pos_test}, neg={neg_test})"
    )

    logger.info("Selected training algorithm: %s", config.algorithm)

    if config.algorithm == "logreg":
        pipeline = _train_logistic_regression(train_set, config)
        model_output_path = config.output_path
        joblib.dump(pipeline, model_output_path)
        logger.info("Persisted TF-IDF + Logistic Regression pipeline to %s", model_output_path)

        texts_test = [_prepare_text(record) for record in test_set]
        y_true = [int(record.get("label", 0)) for record in test_set]
        y_scores = pipeline.predict_proba(texts_test)[:, 1].tolist()

        metrics, roc_data, pr_data = _compute_metrics_from_scores(y_true, y_scores)
        logger.info("Evaluation metrics:\n%s", _format_metrics(metrics).strip())

        metadata = {
            "algorithm": "tfidf_logreg",
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

        # Backwards compatible metadata for downstream tooling
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

    # Warning su overfitting
    if metrics['accuracy'] > 0.95:
        logger.warning(
            "⚠️  Very high accuracy (%.3f) may indicate overfitting. "
            "Consider expanding the dataset or reviewing test/train split.",
            metrics['accuracy']
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
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    cfg = TrainingConfig.from_args(args)
    train(cfg)