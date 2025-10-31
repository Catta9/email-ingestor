from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

from libs.models import ProcessedMessage
from libs.services.ingestion_runner import IngestionEvent, IngestionRunner


load_dotenv()


class StructuredFormatter(logging.Formatter):
    """Formatter JSON per log strutturati."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for attr in ["imap_uid", "message_id", "from_domain", "contact_id", "esito"]:
            if hasattr(record, attr):
                log_obj[attr] = getattr(record, attr)

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging() -> logging.Logger:
    log_format = os.getenv("LOG_FORMAT", "json").lower()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    handler = logging.StreamHandler(sys.stdout)

    if log_format == "json":
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = [handler]

    return logging.getLogger(__name__)


logger = setup_logging()

_RUNNER = IngestionRunner()
_RULE_BASED_SCORER = None
_ML_CLASSIFIER = None
_ML_AVAILABLE = True


def _sync_runner_cache_from_globals() -> None:
    _RUNNER._rule_based_scorer = _RULE_BASED_SCORER
    _RUNNER._ml_classifier = _ML_CLASSIFIER
    _RUNNER._ml_available = _ML_AVAILABLE


def _sync_globals_from_runner() -> None:
    global _RULE_BASED_SCORER, _ML_CLASSIFIER, _ML_AVAILABLE
    _RULE_BASED_SCORER = _RUNNER._rule_based_scorer
    _ML_CLASSIFIER = _RUNNER._ml_classifier
    _ML_AVAILABLE = _RUNNER._ml_available


def allowed_sender(headers: dict[str, str]) -> tuple[bool, str | None]:
    return IngestionRunner._allowed_sender(headers)


def matches_lead_keywords(
    headers: dict[str, str],
    body: str,
    *,
    return_details: bool = False,
) -> bool | tuple[bool, float, str]:
    _sync_runner_cache_from_globals()
    is_lead, score, confidence = _RUNNER._matches_lead_keywords(headers, body)
    _sync_globals_from_runner()
    if return_details:
        return is_lead, score, confidence
    return is_lead


def _mark_sender_disallowed(
    db,
    *,
    message_id: str | None,
    uid_str: str | None,
    from_domain: str | None,
    reason: str,
) -> None:
    _RUNNER._mark_sender_disallowed(
        db,
        message_id=message_id,
        uid_str=uid_str,
        from_domain=from_domain,
        reason=reason,
    )


def log_event_to_logger(event: IngestionEvent) -> None:
    event_type = event.get("type")
    message = event.get("message", "")
    data: dict[str, Any] = event.get("data", {}) or {}

    if event_type == "run_failed":
        logger.error(message, extra={**data, "esito": "error"})
    elif event_type == "lead_created":
        logger.info(message, extra={**data, "esito": "lead_created"})
    elif event_type == "email_skipped":
        logger.info(message, extra={**data, "esito": "skipped"})
    elif event_type == "run_completed":
        logger.info(message, extra={**data, "esito": "summary"})
    else:
        logger.info(message, extra=data)


def main() -> None:
    _sync_runner_cache_from_globals()
    try:
        _RUNNER.run(event_callback=log_event_to_logger)
    finally:
        _sync_globals_from_runner()


if __name__ == "__main__":
    main()
