"""Compatibility helpers for optional third-party dependencies.

This module provides lightweight fallbacks for external packages that are
optional at runtime but required by parts of the project.  When the optional
packages are not installed (for example in constrained CI environments) these
fallbacks mimic a tiny subset of the original API so the application and tests
continue to run.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Tuple


# ---------------------------------------------------------------------------
# joblib compatibility
# ---------------------------------------------------------------------------
try:  # pragma: no cover - only executed when dependency is available
    import joblib as joblib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed in minimal envs

    class _JoblibFallback:
        """Very small subset of :mod:`joblib` using :mod:`pickle` under the hood."""

        def load(self, filename):
            with open(filename, "rb") as handle:
                return pickle.load(handle)

        def dump(self, value, filename):
            with open(filename, "wb") as handle:
                pickle.dump(value, handle)

    joblib = _JoblibFallback()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# prometheus_client compatibility
# ---------------------------------------------------------------------------
try:  # pragma: no cover - dependency available
    import prometheus_client as prometheus_client  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed when dependency missing

    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    @dataclass
    class _Sample:
        name: str
        labels: Dict[str, str]
        value: float

    class _MetricFamily:
        """Container mimicking prometheus client's Metric family."""

        def __init__(self, name: str, documentation: str, samples: Iterable[_Sample]):
            self.name = name
            self.documentation = documentation
            self.samples = list(samples)

    class CollectorRegistry:
        """Minimal CollectorRegistry implementation."""

        def __init__(self, auto_describe: bool = True):
            self.auto_describe = auto_describe
            self._metrics: list[_BaseMetric] = []

        def register(self, metric: "_BaseMetric") -> None:
            self._metrics.append(metric)

        def collect(self) -> Iterator[_MetricFamily]:
            for metric in list(self._metrics):
                yield metric._collect()

    class _BaseMetric:
        def __init__(
            self,
            name: str,
            documentation: str,
            labelnames: Tuple[str, ...] = (),
            registry: CollectorRegistry | None = None,
        ) -> None:
            self._name = name
            self._documentation = documentation
            self._labelnames = tuple(labelnames)
            self._registry = registry or CollectorRegistry()
            self._series: Dict[Tuple[str, ...], "_SeriesBase"] = {}
            self._registry.register(self)

        def labels(self, *args, **kwargs):
            if args and kwargs:
                raise TypeError("Cannot use both args and kwargs for labels")
            if args:
                if len(args) != len(self._labelnames):
                    raise ValueError("Incorrect number of labels")
                label_values = tuple(str(v) for v in args)
                labels = dict(zip(self._labelnames, label_values))
            else:
                labels = {name: str(kwargs.get(name, "")) for name in self._labelnames}
                label_values = tuple(labels[name] for name in self._labelnames)
            series = self._series.get(label_values)
            if series is None:
                series = self._create_series(labels)
                self._series[label_values] = series
            return series

        def clear(self) -> None:
            self._series.clear()

        # The subclasses override these methods
        def _create_series(self, labels: Dict[str, str]):
            raise NotImplementedError

        def _collect(self) -> _MetricFamily:
            raise NotImplementedError

    class _SeriesBase:
        def __init__(self, labels: Dict[str, str]):
            self.labels = labels
            self.value = 0.0

    class _CounterSeries(_SeriesBase):
        def inc(self, amount: float = 1.0) -> None:
            if amount < 0:
                raise ValueError("Counters can only be incremented by non-negative amounts")
            self.value += float(amount)

    class _GaugeSeries(_SeriesBase):
        def set(self, value: float) -> None:
            self.value = float(value)

    class Counter(_BaseMetric):
        def _create_series(self, labels: Dict[str, str]) -> _CounterSeries:
            return _CounterSeries(labels)

        def _collect(self) -> _MetricFamily:
            samples = (
                _Sample(f"{self._name}_total", series.labels, series.value)
                for series in self._series.values()
            )
            return _MetricFamily(self._name, self._documentation, samples)

    class Gauge(_BaseMetric):
        def _create_series(self, labels: Dict[str, str]) -> _GaugeSeries:
            return _GaugeSeries(labels)

        def _collect(self) -> _MetricFamily:
            samples = (
                _Sample(self._name, series.labels, series.value)
                for series in self._series.values()
            )
            return _MetricFamily(self._name, self._documentation, samples)

    def generate_latest(registry: CollectorRegistry) -> bytes:
        lines = []
        for metric in registry.collect():
            for sample in metric.samples:
                if sample.labels:
                    labels = ",".join(
                        f'{key}="{value}"' for key, value in sorted(sample.labels.items())
                    )
                    lines.append(f"{sample.name}{{{labels}}} {sample.value}")
                else:
                    lines.append(f"{sample.name} {sample.value}")
        return ("\n".join(lines) + "\n").encode("utf-8")

    class _PrometheusModule:
        CONTENT_TYPE_LATEST = CONTENT_TYPE_LATEST
        CollectorRegistry = CollectorRegistry
        Counter = Counter
        Gauge = Gauge
        generate_latest = staticmethod(generate_latest)

    prometheus_client = _PrometheusModule()  # type: ignore[assignment]

__all__ = ["joblib", "prometheus_client"]
