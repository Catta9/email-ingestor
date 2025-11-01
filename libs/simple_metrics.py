"""Pure-Python fallbacks for a subset of :mod:`sklearn.metrics`."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


class MetricsError(ValueError):
    """Raised when metrics cannot be computed (e.g. missing positive samples)."""


def _to_lists(y_true: Iterable[float], y_scores: Iterable[float]) -> Tuple[List[float], List[float]]:
    y_true_list = [float(v) for v in y_true]
    y_scores_list = [float(v) for v in y_scores]
    if len(y_true_list) != len(y_scores_list):
        raise MetricsError("y_true and y_scores must have the same length")
    return y_true_list, y_scores_list


def _binary_clf_curve(y_true: Sequence[float], y_scores: Sequence[float]) -> Tuple[List[float], List[float], List[float]]:
    order = sorted(range(len(y_scores)), key=y_scores.__getitem__, reverse=True)
    tps: List[float] = []
    fps: List[float] = []
    thresholds: List[float] = []
    tp = fp = 0.0
    prev_score = None

    for idx in order:
        score = y_scores[idx]
        label = y_true[idx]
        if prev_score is None or score != prev_score:
            thresholds.append(score)
            tps.append(tp)
            fps.append(fp)
            prev_score = score
        if label == 1:
            tp += 1.0
        else:
            fp += 1.0
        tps[-1] = tp
        fps[-1] = fp

    return fps, tps, thresholds


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _trapz(y: Sequence[float], x: Sequence[float]) -> float:
    return sum((x1 - x0) * (y0 + y1) / 2.0 for (x0, x1, y0, y1) in zip(x[:-1], x[1:], y[:-1], y[1:]))


def roc_curve(y_true: Iterable[float], y_scores: Iterable[float]):
    y_true_list, y_scores_list = _to_lists(y_true, y_scores)
    positives = sum(1.0 for v in y_true_list if v == 1.0)
    negatives = sum(1.0 for v in y_true_list if v == 0.0)
    if positives == 0 or negatives == 0:
        raise MetricsError("Both positive and negative samples are required to compute ROC")

    fps, tps, thresholds = _binary_clf_curve(y_true_list, y_scores_list)

    fpr = [0.0]
    tpr = [0.0]
    final_thresholds = [thresholds[0] + 1.0 if thresholds else 1.0]

    for fp, tp, threshold in zip(fps, tps, thresholds):
        fpr.append(_safe_div(fp, negatives))
        tpr.append(_safe_div(tp, positives))
        final_thresholds.append(threshold)

    return fpr, tpr, final_thresholds


def roc_auc_score(y_true: Iterable[float], y_scores: Iterable[float]) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return _trapz(tpr, fpr)


def precision_recall_curve(y_true: Iterable[float], y_scores: Iterable[float]):
    y_true_list, y_scores_list = _to_lists(y_true, y_scores)
    positives = sum(1.0 for v in y_true_list if v == 1.0)
    if positives == 0:
        raise MetricsError("At least one positive sample is required to compute precision/recall")

    fps, tps, thresholds = _binary_clf_curve(y_true_list, y_scores_list)

    precision: List[float] = [1.0]
    recall: List[float] = [0.0]

    for fp, tp in zip(fps, tps):
        precision.append(_safe_div(tp, tp + fp))
        recall.append(_safe_div(tp, positives))

    return precision, recall, thresholds
