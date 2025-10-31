from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from libs.ml_classifier import LeadMLClassifier, LeadModelConfig


logger = logging.getLogger(__name__)


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid JSON at line {idx}: {exc}") from exc
            if "label" not in record:
                raise ValueError(f"Missing 'label' field at line {idx}")
            records.append(record)
    if not records:
        raise ValueError(f"Dataset {path} is empty")
    return records


def _compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], dict[str, list[float]], dict[str, list[float]]]:
    y_pred = (y_scores >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_scores)

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
        "threshold": threshold,
    }

    roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist()}
    pr_data = {
        "precision": pr_precision.tolist(),
        "recall": pr_recall.tolist(),
        "thresholds": pr_thresholds.tolist(),
    }
    return metrics, roc_data, pr_data


def evaluate(dataset_path: Path, model_path: Path | None, output_dir: Path) -> Path:
    records = _load_dataset(dataset_path)

    config = LeadModelConfig.from_env()
    if model_path is not None:
        config.model_path = model_path
    classifier = LeadMLClassifier(config)

    y_true = []
    y_scores = []

    for record in records:
        headers = {"Subject": record.get("subject", "")}
        body = record.get("body", "")
        score = classifier.score(headers, body)
        y_true.append(int(record.get("label", 0)))
        y_scores.append(float(score))

    y_true_array = np.asarray(y_true)
    y_scores_array = np.asarray(y_scores)
    threshold = classifier.decision_threshold

    metrics, roc_data, pr_data = _compute_metrics(y_true_array, y_scores_array, threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "evaluation_metrics.json"
    curves_path = output_dir / "evaluation_curves.json"

    metrics_path.write_text(json.dumps({"metrics": metrics}, ensure_ascii=False, indent=2), encoding="utf-8")
    curves_path.write_text(
        json.dumps({"roc": roc_data, "precision_recall": pr_data}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Saved evaluation metrics to %s", metrics_path)
    logger.info("Saved ROC and precision-recall curves to %s", curves_path)

    return metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained lead classification model")
    parser.add_argument(
        "--dataset",
        default="datasets/lead_training.jsonl",
        help="Path to the JSONL dataset used for evaluation",
    )
    parser.add_argument(
        "--model",
        help="Optional explicit path to the trained model (joblib or json). Defaults to LEAD_MODEL_PATH",
    )
    parser.add_argument(
        "--output",
        default="artifacts",
        help="Directory where evaluation artifacts will be stored",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. INFO, DEBUG)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    dataset_path = Path(args.dataset).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve() if args.model else None
    output_dir = Path(args.output).expanduser().resolve()

    evaluate(dataset_path, model_path, output_dir)


if __name__ == "__main__":
    main()
