from __future__ import annotations

import logging
import os
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any, Callable, Dict, Iterable, List

from sqlalchemy import select

from libs.db import SessionLocal, init_db
from libs.email_utils import (
    extract_headers,
    fetch_message,
    get_imap_client,
    get_text_body,
    search_since_days,
)
from libs.ingestor import extract_sender_domain, process_incoming_email
from libs.lead_classifier import LeadRelevanceScorer
from libs.lead_storage import ExcelLeadWriter
from libs.ml_classifier import LeadMLClassifier, ModelNotAvailableError
from libs.models import ProcessedMessage
from libs.notifier import EmailNotifier


logger = logging.getLogger(__name__)


class IngestionEvent(Dict[str, Any]):
    """Simple mapping describing an ingestion lifecycle event."""


@dataclass
class IngestionStats:
    processed_count: int = 0
    lead_count: int = 0
    skipped_count: int = 0
    total: int = 0


class IngestionRunner:
    """Reusable ingestion runner that emits structured events."""

    def __init__(self) -> None:
        self._rule_based_scorer: LeadRelevanceScorer | None = None
        self._ml_classifier: LeadMLClassifier | None = None
        self._ml_available: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, event_callback: Callable[[IngestionEvent], None] | None = None) -> IngestionStats:
        """Run the ingestion and emit events through ``event_callback``."""

        if event_callback is None:
            event_callback = lambda event: None

        def emit(event_type: str, message: str, **data: Any) -> None:
            payload: IngestionEvent = {"type": event_type, "message": message}
            if data:
                payload["data"] = data
            event_callback(payload)

        init_db()
        folder = os.getenv("IMAP_FOLDER", "INBOX")
        since_days = int(os.getenv("IMAP_SEARCH_SINCE_DAYS", "7"))

        excel_writer = ExcelLeadWriter.from_env()
        notifier = EmailNotifier.from_env()

        client = get_imap_client()
        stats = IngestionStats()
        uids: Iterable[int] = []

        emit(
            "run_started",
            "Inizio ingestione delle email",
            folder=folder,
            since_days=since_days,
        )
        logger.info(
            "Starting email ingestion",
            extra={
                "folder": folder,
                "since_days": since_days,
                "strategy": os.getenv("LEAD_CLASSIFIER_STRATEGY", "rule_based"),
            },
        )

        try:
            uids = search_since_days(client, folder, since_days)
            stats.total = len(uids)
            emit(
                "run_progress",
                "Trovati %d messaggi da processare" % stats.total,
                total=stats.total,
            )

            for uid in uids:
                message = fetch_message(client, uid)
                headers = extract_headers(message)
                msg_id = (headers.get("Message-ID") or "").strip()
                uid_str = str(uid)
                from_domain = extract_sender_domain(headers)
                log_context = {
                    "imap_uid": uid_str,
                    "message_id": msg_id or uid_str,
                    "from_domain": from_domain,
                }

                with SessionLocal() as db:
                    if self._already_processed(db, msg_id, uid_str):
                        stats.skipped_count += 1
                        emit(
                            "email_skipped",
                            "Email già processata (UID %s)" % uid_str,
                            reason="duplicate",
                            **log_context,
                        )
                        logger.debug(
                            "Email already processed",
                            extra={**log_context, "esito": "duplicate"},
                        )
                        continue

                    is_allowed, reason = self._allowed_sender(headers)
                    if not is_allowed:
                        skip_reason = reason or f"Sender not allowed: {headers.get('From')}"
                        self._mark_sender_disallowed(
                            db,
                            message_id=msg_id or None,
                            uid_str=uid_str,
                            from_domain=from_domain,
                            reason=skip_reason,
                        )
                        stats.skipped_count += 1
                        emit(
                            "email_skipped",
                            skip_reason,
                            reason="not_allowed",
                            **log_context,
                        )
                        continue

                    body = get_text_body(message)
                    is_lead, score, confidence = self._matches_lead_keywords(
                        headers,
                        body,
                    )

                    if not is_lead:
                        skip_reason = (
                            f"Missing lead keywords (score={score:.2f}, conf={confidence})"
                        )
                        self._mark_sender_disallowed(
                            db,
                            message_id=msg_id or None,
                            uid_str=uid_str,
                            from_domain=from_domain,
                            reason=skip_reason,
                        )
                        stats.skipped_count += 1
                        emit(
                            "email_skipped",
                            skip_reason,
                            reason="not_lead",
                            score=score,
                            confidence=confidence,
                            **log_context,
                        )
                        logger.debug(
                            "Email rejected by lead classifier",
                            extra={
                                **log_context,
                                "esito": "not_lead",
                                "score": score,
                                "confidence": confidence,
                            },
                        )
                        continue

                    received_at = self._parse_received_at(headers)
                    result = process_incoming_email(
                        db,
                        headers=headers,
                        body=body,
                        imap_uid=uid,
                        received_at=received_at,
                    )

                    stats.processed_count += 1
                    contact_id = result.get("contact_id")
                    result_context: Dict[str, Any] = {**log_context}
                    if contact_id is not None:
                        result_context["contact_id"] = contact_id

                    if result["status"] == "processed":
                        if result.get("created"):
                            stats.lead_count += 1
                            lead_payload = {
                                "inserted_at": datetime.utcnow(),
                                "email": result["extracted"].get("email"),
                                "first_name": result["extracted"].get("first_name"),
                                "last_name": result["extracted"].get("last_name"),
                                "company": result["extracted"].get("org"),
                                "phone": result["extracted"].get("phone"),
                                "subject": result.get("subject"),
                                "received_at": result.get("received_at"),
                                "notes": result.get("body_excerpt"),
                            }
                            excel_row = None
                            if excel_writer:
                                excel_row = excel_writer.append(lead_payload)
                            if notifier:
                                with suppress(Exception):
                                    payload = {
                                        **{
                                            k: v
                                            for k, v in lead_payload.items()
                                            if isinstance(v, str)
                                        }
                                    }
                                    received_at_val = lead_payload.get("received_at")
                                    if isinstance(received_at_val, datetime):
                                        payload["received_at"] = received_at_val.isoformat()
                                    elif received_at_val is not None:
                                        payload["received_at"] = str(received_at_val)
                                    notifier.send_new_lead(payload)
                                    if excel_writer and excel_row:
                                        notifier.send_excel_update(
                                            payload,
                                            workbook_path=str(excel_writer.path),
                                            row_number=excel_row,
                                        )

                            emit(
                                "lead_created",
                                "Nuovo lead creato",
                                contact_id=contact_id,
                                extracted=result.get("extracted", {}),
                                **log_context,
                            )

                        emit(
                            "email_processed",
                            "Email elaborata con successo",
                            lead_score=score,
                            confidence=confidence,
                            **result_context,
                        )
                        logger.info(
                            "Email ingested successfully",
                            extra={
                                **result_context,
                                "esito": "ingested",
                                "is_new": result.get("created", False),
                                "lead_score": score,
                                "confidence": confidence,
                            },
                        )
                    else:
                        stats.skipped_count += 1
                        reason = result.get("reason", "unknown")
                        emit(
                            "email_skipped",
                            f"Email saltata: {reason}",
                            reason=reason,
                            **result_context,
                        )
                        logger.info(
                            "Email skipped by ingestion",
                            extra={
                                **result_context,
                                "esito": "skipped",
                                "reason": reason,
                            },
                        )

                emit(
                    "run_progress",
                    "Avanzamento ingestione",
                    processed=stats.processed_count,
                    skipped=stats.skipped_count,
                    leads=stats.lead_count,
                    total=stats.total,
                )

        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Ingestion failed with exception",
                exc_info=True,
                extra={"esito": "error"},
            )
            emit(
                "run_failed",
                f"Errore nell'ingestione: {exc}",
            )
            raise
        finally:
            with suppress(Exception):
                client.logout()

            emit(
                "run_completed",
                "Ingestione completata",
                processed=stats.processed_count,
                new_leads=stats.lead_count,
                skipped=stats.skipped_count,
                total=stats.total,
            )
            logger.info(
                "Ingestion completed",
                extra={
                    "processed": stats.processed_count,
                    "new_leads": stats.lead_count,
                    "skipped": stats.skipped_count,
                    "total": stats.total,
                },
            )

        return stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _allowed_sender(headers: Dict[str, str]) -> tuple[bool, str | None]:
        raw_from = headers.get("From") or ""
        _, email_address = parseaddr(raw_from)
        email_address = email_address.strip()

        if not email_address or "@" not in email_address:
            return False, f"Invalid sender address: {raw_from!r}"

        domain = email_address.rsplit("@", 1)[1].lower()

        allowed = os.getenv("ALLOWED_SENDER_DOMAINS", "").strip()
        if not allowed:
            return True, None

        allowed_domains = {d.strip().lower() for d in allowed.split(",") if d.strip()}
        if domain in allowed_domains:
            return True, None

        return False, f"Sender domain not allowed: {domain}"

    @staticmethod
    def _parse_received_at(headers: Dict[str, str]) -> datetime | None:
        raw_date = headers.get("Date") or headers.get("date")
        if not raw_date:
            return None
        try:
            return parsedate_to_datetime(raw_date)
        except Exception:
            return None

    def _already_processed(self, db, message_id: str | None, uid_str: str | None) -> bool:
        if message_id:
            processed = (
                db.execute(
                    select(ProcessedMessage).where(ProcessedMessage.message_id == message_id)
                ).scalar_one_or_none()
            )
            if processed is not None:
                return True
        if uid_str:
            processed = (
                db.execute(
                    select(ProcessedMessage).where(ProcessedMessage.imap_uid == uid_str)
                ).scalar_one_or_none()
            )
            if processed is not None:
                return True
        return False

    def _mark_sender_disallowed(
        self,
        db,
        *,
        message_id: str | None,
        uid_str: str | None,
        from_domain: str | None,
        reason: str,
    ) -> None:
        identifier = message_id or uid_str
        logger.info(
            "Skipping email due to sender restrictions: %s",
            reason,
            extra={
                "imap_uid": uid_str,
                "message_id": identifier,
                "from_domain": from_domain,
                "esito": "skipped",
            },
        )
        stored_message_id = message_id or (f"uid:{uid_str}" if uid_str else None)
        if stored_message_id is not None:
            db.add(ProcessedMessage(message_id=stored_message_id, imap_uid=uid_str))
            db.commit()

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    def _get_rule_based_scorer(self) -> LeadRelevanceScorer:
        if self._rule_based_scorer is None:
            self._rule_based_scorer = LeadRelevanceScorer.from_env()
        return self._rule_based_scorer

    def _get_ml_classifier(self) -> LeadMLClassifier | None:
        if not self._ml_available:
            return None
        if self._ml_classifier is None:
            try:
                self._ml_classifier = LeadMLClassifier.from_env()
            except ModelNotAvailableError as exc:
                logger.warning("ML classifier unavailable: %s", exc)
                self._ml_available = False
                return None
        return self._ml_classifier

    def _use_ml_strategy(self) -> bool:
        strategy = os.getenv("LEAD_CLASSIFIER_STRATEGY", "rule_based").strip().lower()
        return strategy == "ml" or strategy == "hybrid"

    def _use_hybrid_strategy(self) -> bool:
        strategy = os.getenv("LEAD_CLASSIFIER_STRATEGY", "rule_based").strip().lower()
        return strategy == "hybrid"

    def _matches_lead_keywords(
        self,
        headers: Dict[str, str],
        body: str,
    ) -> tuple[bool, float, str]:
        classifier = self._get_ml_classifier() if self._use_ml_strategy() else None
        score: float = 0.0
        confidence = "rule_based"

        if classifier is not None:
            try:
                score, confidence = classifier.score_with_confidence(headers, body)
            except ModelNotAvailableError as exc:
                logger.warning("ML classifier unavailable during scoring: %s", exc)
                classifier = None
                self._ml_available = False

        if classifier is not None:
            is_lead = score >= classifier.config.threshold
            method_confidence = confidence

            if not is_lead and self._use_hybrid_strategy():
                scorer = self._get_rule_based_scorer()
                rule_score = scorer.score(headers, body)
                rule_is_lead = scorer.is_relevant(headers, body)
                if rule_is_lead:
                    score = rule_score
                    method_confidence = "rule_fallback"
                    is_lead = True

            return is_lead, score, method_confidence

        scorer = self._get_rule_based_scorer()
        score = scorer.score(headers, body)
        is_lead = scorer.is_relevant(headers, body)
        return is_lead, score, "rule_based"
