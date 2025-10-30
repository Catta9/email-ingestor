from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[\w']+")


@dataclass
class TrainingConfig:
    dataset_path: Path
    output_path: Path
    test_size: float = 0.2
    random_state: int = 42

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        dataset_path = Path(args.dataset).expanduser().resolve()
        output_dir = Path(args.output).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "lead_classifier.json"
        return cls(
            dataset_path=dataset_path,
            output_path=output_path,
            test_size=args.test_size,
            random_state=args.random_state,
        )


# ---------------------------------------------------------------------------
def _load_records(path: Path) -> list[dict[str, str]]:
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
            records.append(record)
    if not records:
        raise ValueError("Dataset is empty")
    return records


def _prepare_text(record: dict[str, str]) -> str:
    subject = (record.get("subject") or "").strip()
    body = (record.get("body") or "").strip()
    if subject and body:
        return f"{subject}\n{body}"
    return subject or body


def _tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token]


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


def _count_tokens(records: Iterable[dict[str, str]]) -> tuple[dict[str, int], dict[str, int]]:
    counts_positive: dict[str, int] = {}
    counts_negative: dict[str, int] = {}
    for record in records:
        tokens = _tokenize(_prepare_text(record))
        target = counts_positive if int(record.get("label", 0)) == 1 else counts_negative
        for token in tokens:
            target[token] = target.get(token, 0) + 1
    return counts_positive, counts_negative


def _train_naive_bayes(train_set: list[dict[str, str]]):
    pos_counts, neg_counts = _count_tokens(train_set)
    total_pos_tokens = sum(pos_counts.values())
    total_neg_tokens = sum(neg_counts.values())
    vocab = set(pos_counts) | set(neg_counts)

    num_pos = sum(1 for record in train_set if int(record.get("label", 0)) == 1)
    num_neg = len(train_set) - num_pos
    total_docs = max(1, len(train_set))

    model = {
        "version": 1,
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
    return model


def _predict_proba(model: dict, text: str) -> float:
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

    max_log = max(log_pos, log_neg)
    pos_prob = math.exp(log_pos - max_log)
    neg_prob = math.exp(log_neg - max_log)
    total = pos_prob + neg_prob
    if total == 0:
        return 0.0
    return pos_prob / total


def _evaluate(model: dict, records: Iterable[dict[str, str]]) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for record in records:
        text = _prepare_text(record)
        proba = _predict_proba(model, text)
        predicted = 1 if proba >= 0.5 else 0
        actual = int(record.get("label", 0))
        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 1 and actual == 0:
            fp += 1
        elif predicted == 0 and actual == 0:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _format_metrics(metrics: dict[str, float]) -> str:
    return (
        "Accuracy: {accuracy:.3f}\n"
        "Precision: {precision:.3f}\n"
        "Recall: {recall:.3f}\n"
        "F1-score: {f1:.3f}\n"
        "TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}\n"
    ).format(**metrics)


def train(config: TrainingConfig) -> None:
    records = _load_records(config.dataset_path)
    train_set, test_set = _split_dataset(records, config.test_size, config.random_state)
    logger.info("Loaded %d records (train=%d, test=%d)", len(records), len(train_set), len(test_set))

    model = _train_naive_bayes(train_set)
    metrics = _evaluate(model, test_set)
    logger.info("Evaluation metrics:\n%s", _format_metrics(metrics).strip())

    config.output_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Model persisted to %s", config.output_path)

    metrics_path = config.output_path.with_suffix(".metrics.txt")
    metrics_path.write_text(_format_metrics(metrics), encoding="utf-8")
    logger.info("Saved evaluation metrics to %s", metrics_path)

    default_store = Path(os.getenv("LEAD_MODEL_PATH", "model_store/lead_classifier.json"))
    if default_store.resolve() != config.output_path.resolve():
        default_store.parent.mkdir(parents=True, exist_ok=True)
        default_store.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Copied model to default store: %s", default_store)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
        "--log-level",
        default="INFO",
        help="Logging level (e.g. INFO, DEBUG)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper())
    cfg = TrainingConfig.from_args(args)
    train(cfg)
