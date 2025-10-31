"""Prometheus metrics utilities for the email ingestor."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, generate_latest

_REGISTRY = CollectorRegistry(auto_describe=True)

_EMAILS_PROCESSED = Counter(
    "email_ingestor_processed_total",
    "Numero di email elaborate con successo",
    labelnames=("folder", "domain"),
    registry=_REGISTRY,
)

_LEADS_CREATED = Counter(
    "email_ingestor_leads_total",
    "Numero di lead creati dal processo di ingestione",
    labelnames=("folder", "domain"),
    registry=_REGISTRY,
)

_INGESTION_ERRORS = Counter(
    "email_ingestor_errors_total",
    "Numero di errori riscontrati durante l'ingestione",
    labelnames=("folder", "domain"),
    registry=_REGISTRY,
)

_EMAILS_DISCOVERED = Gauge(
    "email_ingestor_run_discovered_messages",
    "Numero totale di email individuate nell'ultima esecuzione",
    labelnames=("folder",),
    registry=_REGISTRY,
)


def _sanitize_folder(folder: str | None) -> str:
    return (folder or "unknown").strip() or "unknown"


def _sanitize_domain(domain: str | None) -> str:
    if not domain:
        return "unknown"
    domain = domain.strip().lower()
    return domain or "unknown"


def get_registry() -> CollectorRegistry:
    """Return the metrics registry used by the application."""

    return _REGISTRY


def record_email_processed(*, folder: str | None, domain: str | None) -> None:
    """Increment the counter of processed emails for the given labels."""

    _EMAILS_PROCESSED.labels(
        folder=_sanitize_folder(folder), domain=_sanitize_domain(domain)
    ).inc()


def record_lead_created(*, folder: str | None, domain: str | None) -> None:
    """Increment the counter of created leads for the given labels."""

    _LEADS_CREATED.labels(
        folder=_sanitize_folder(folder), domain=_sanitize_domain(domain)
    ).inc()


def record_ingestion_error(*, folder: str | None, domain: str | None) -> None:
    """Increment the counter of ingestion errors for the given labels."""

    _INGESTION_ERRORS.labels(
        folder=_sanitize_folder(folder), domain=_sanitize_domain(domain)
    ).inc()


def set_discovered_messages(*, folder: str | None, total: int) -> None:
    """Track the number of messages discovered at the beginning of a run."""

    _EMAILS_DISCOVERED.labels(folder=_sanitize_folder(folder)).set(float(total))


def metrics_snapshot(folder: str | None = None) -> Dict[str, Dict[str, float]]:
    """Return an aggregated snapshot of the key metrics.

    The snapshot structure is::

        {
            "emails_processed": {"total": 10, "by_domain": {"example.com": 5}},
            "leads_created": {"total": 2, "by_domain": {"example.com": 1}},
            "ingestion_errors": {...},
            "emails_discovered": {"total": 42},
        }
    """

    folder_label = _sanitize_folder(folder) if folder is not None else None

    def _aggregate_counter(counter: Counter) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        per_domain: Dict[str, float] = defaultdict(float)
        for metric in counter.collect():
            for sample in metric.samples:
                # Ignore metadata samples automatically generated for counters
                if not sample.name.endswith("_total"):
                    continue
                labels = sample.labels or {}
                if folder_label is not None and labels.get("folder") != folder_label:
                    continue
                domain = labels.get("domain", "unknown")
                per_domain[domain] += float(sample.value)
                total += float(sample.value)
        return total, dict(per_domain)

    def _aggregate_gauge(gauge: Gauge) -> float:
        value = 0.0
        for metric in gauge.collect():
            for sample in metric.samples:
                labels = sample.labels or {}
                if folder_label is not None and labels.get("folder") != folder_label:
                    continue
                value = float(sample.value)
        return value

    processed_total, processed_by_domain = _aggregate_counter(_EMAILS_PROCESSED)
    leads_total, leads_by_domain = _aggregate_counter(_LEADS_CREATED)
    errors_total, errors_by_domain = _aggregate_counter(_INGESTION_ERRORS)
    discovered_total = _aggregate_gauge(_EMAILS_DISCOVERED)

    snapshot = {
        "emails_processed": {
            "total": processed_total,
            "by_domain": processed_by_domain,
        },
        "leads_created": {
            "total": leads_total,
            "by_domain": leads_by_domain,
        },
        "ingestion_errors": {
            "total": errors_total,
            "by_domain": errors_by_domain,
        },
        "emails_discovered": {"total": discovered_total},
    }
    return snapshot


def render_metrics() -> bytes:
    """Render the metrics registry in Prometheus exposition format."""

    return generate_latest(_REGISTRY)


def reset_metrics() -> None:
    """Clear all recorded metric series. Intended for tests."""

    _EMAILS_PROCESSED.clear()
    _LEADS_CREATED.clear()
    _INGESTION_ERRORS.clear()
    _EMAILS_DISCOVERED.clear()


__all__ = [
    "CONTENT_TYPE_LATEST",
    "metrics_snapshot",
    "record_email_processed",
    "record_ingestion_error",
    "record_lead_created",
    "render_metrics",
    "reset_metrics",
    "set_discovered_messages",
    "get_registry",
]
