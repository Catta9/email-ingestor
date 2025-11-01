"""Minimal Prometheus client used when the real dependency is unavailable."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


@dataclass
class _Sample:
    name: str
    labels: Dict[str, str]
    value: float


@dataclass
class _Metric:
    name: str
    samples: list[_Sample]


class CollectorRegistry:
    def __init__(self, auto_describe: bool = False) -> None:  # noqa: ARG002 - keep signature
        self._metrics: list[Tuple[str, _BaseMetric]] = []

    def register(self, metric: "_BaseMetric") -> None:
        self._metrics.append((metric.name, metric))

    def collect(self) -> Iterable[_Metric]:
        for _, metric in self._metrics:
            yield from metric.collect()


class _BaseMetric:
    def __init__(self, name: str, documentation: str, labelnames: Tuple[str, ...], registry: CollectorRegistry):
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames
        self._values: Dict[Tuple[str, ...], float] = defaultdict(float)
        registry.register(self)

    def labels(self, **labels: str) -> "_MetricHandle":
        label_tuple = tuple(labels.get(key, "") for key in self.labelnames)
        return _MetricHandle(self, label_tuple)

    def _format_labels(self, label_tuple: Tuple[str, ...]) -> Dict[str, str]:
        return {name: value for name, value in zip(self.labelnames, label_tuple)}

    def _samples(self) -> list[_Sample]:
        samples = []
        for label_tuple, value in self._values.items():
            labels = self._format_labels(label_tuple)
            sample_name = self.name if isinstance(self, Gauge) else f"{self.name}_total"
            samples.append(_Sample(sample_name, labels, float(value)))
        return samples

    def collect(self) -> Iterable[_Metric]:
        yield _Metric(self.name, self._samples())

    def clear(self) -> None:
        self._values.clear()


class Counter(_BaseMetric):
    def labels(self, **labels: str) -> "_MetricHandle":  # type: ignore[override]
        handle = super().labels(**labels)
        self._values.setdefault(handle._label_tuple, 0.0)
        return handle


class Gauge(_BaseMetric):
    pass


class _MetricHandle:
    def __init__(self, metric: _BaseMetric, label_tuple: Tuple[str, ...]) -> None:
        self._metric = metric
        self._label_tuple = label_tuple

    def inc(self, amount: float = 1.0) -> None:
        if not isinstance(self._metric, Counter):
            raise TypeError("inc is only supported on Counter metrics")
        self._metric._values[self._label_tuple] += float(amount)

    def set(self, value: float) -> None:
        if not isinstance(self._metric, Gauge):
            raise TypeError("set is only supported on Gauge metrics")
        self._metric._values[self._label_tuple] = float(value)


def generate_latest(registry: CollectorRegistry) -> bytes:
    lines = []
    for name, metric in registry._metrics:
        lines.append(f"# HELP {name} {metric.documentation}")
        metric_type = "gauge" if isinstance(metric, Gauge) else "counter"
        lines.append(f"# TYPE {name} {metric_type}")
        for sample in metric._samples():
            if sample.labels:
                labels = ",".join(f"{k}=\"{v}\"" for k, v in sample.labels.items())
                lines.append(f"{sample.name}{{{labels}}} {sample.value}")
            else:
                lines.append(f"{sample.name} {sample.value}")
    return "\n".join(lines).encode("utf-8")
