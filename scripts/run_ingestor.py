from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import suppress
from datetime import datetime
from email.utils import parseaddr, parsedate_to_datetime

from dotenv import load_dotenv
from sqlalchemy import select

from libs.db import SessionLocal, init_db
from libs.models import ProcessedMessage
from libs.email_utils import (
    extract_headers,
    fetch_message,
    get_imap_client,
    get_text_body,
    search_since_days,
)
from libs.ingestor import extract_sender_domain, process_incoming_email
from libs.lead_classifier import LeadRelevanceScorer
from libs.ml_classifier import LeadMLClassifier, ModelNotAvailableError
from libs.lead_storage import ExcelLeadWriter
from libs.notifier import EmailNotifier

load_dotenv()


# Logging strutturato con JSON per facilità parsing
class StructuredFormatter(logging.Formatter):
    """Formatter JSON per log strutturati."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Aggiungi campi extra strutturati
        for attr in ["imap_uid", "message_id", "from_domain", "contact_id", "esito"]:
            if hasattr(record, attr):
                log_obj[attr] = getattr(record, attr)
        
        # Aggiungi exception se presente
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging():
    """Configura logging strutturato."""
    log_format = os.getenv("LOG_FORMAT", "json").lower()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    handler = logging.StreamHandler(sys.stdout)
    
    if log_format == "json":
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = [handler]
    
    return logging.getLogger(__name__)


logger = setup_logging()


def allowed_sender(headers: dict) -> tuple[bool, str | None]:
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


def parse_received_at(headers: dict[str, str]) -> datetime | None:
    raw_date = headers.get("Date") or headers.get("date")
    if not raw_date:
        return None
    try:
        return parsedate_to_datetime(raw_date)
    except Exception:
        return None


_RULE_BASED_SCORER: LeadRelevanceScorer | None = None
_ML_CLASSIFIER: LeadMLClassifier | None = None
_ML_AVAILABLE = True


def _get_rule_based_scorer() -> LeadRelevanceScorer:
    global _RULE_BASED_SCORER
    if _RULE_BASED_SCORER is None:
        _RULE_BASED_SCORER = LeadRelevanceScorer.from_env()
    return _RULE_BASED_SCORER


def _get_ml_classifier() -> LeadMLClassifier | None:
    global _ML_CLASSIFIER, _ML_AVAILABLE
    if not _ML_AVAILABLE:
        return None
    if _ML_CLASSIFIER is None:
        try:
            _ML_CLASSIFIER = LeadMLClassifier.from_env()
        except ModelNotAvailableError as exc:
            logger.warning("ML classifier unavailable: %s", exc)
            _ML_AVAILABLE = False
            return None
    return _ML_CLASSIFIER


def _use_ml_strategy() -> bool:
    strategy = os.getenv("LEAD_CLASSIFIER_STRATEGY", "rule_based").strip().lower()
    return strategy == "ml" or strategy == "hybrid"


def _use_hybrid_strategy() -> bool:
    strategy = os.getenv("LEAD_CLASSIFIER_STRATEGY", "rule_based").strip().lower()
    return strategy == "hybrid"


def matches_lead_keywords(
    headers: dict[str, str],
    body: str,
    *,
    return_details: bool = False,
) -> bool | tuple[bool, float, str]:
    """
    Verifica se l'email è un lead.
    
    Returns:
        Se ``return_details`` è ``True`` restituisce ``(is_lead, score, confidence)``.
        Altrimenti restituisce solo ``is_lead``.
    """
    global _ML_AVAILABLE
    classifier = _get_ml_classifier() if _use_ml_strategy() else None
    score: float = 0.0
    confidence = "rule_based"

    if classifier is not None:
        try:
            score, confidence = classifier.score_with_confidence(headers, body)
        except ModelNotAvailableError as exc:
            logger.warning("ML classifier unavailable during scoring: %s", exc)
            classifier = None
            _ML_AVAILABLE = False

    if classifier is not None:
        is_lead = score >= classifier.config.threshold
        method_confidence = confidence

        if not is_lead and _use_hybrid_strategy():
            scorer = _get_rule_based_scorer()
            rule_score = scorer.score(headers, body)
            rule_is_lead = scorer.is_relevant(headers, body)
            if rule_is_lead:
                score = rule_score
                method_confidence = "rule_fallback"
                is_lead = True

        if return_details:
            return is_lead, score, method_confidence
        return is_lead

    scorer = _get_rule_based_scorer()
    score = scorer.score(headers, body)
    is_lead = scorer.is_relevant(headers, body)
    if return_details:
        return is_lead, score, "rule_based"
    return is_lead


def _already_processed(db, message_id: str | None, uid_str: str | None) -> bool:
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


def main():
    """Ingestor principale con logging migliorato."""
    init_db()
    folder = os.getenv("IMAP_FOLDER", "INBOX")
    since_days = int(os.getenv("IMAP_SEARCH_SINCE_DAYS", "7"))

    excel_writer = ExcelLeadWriter.from_env()
    notifier = EmailNotifier.from_env()

    logger.info(
        "Starting email ingestion",
        extra={
            "folder": folder,
            "since_days": since_days,
            "strategy": os.getenv("LEAD_CLASSIFIER_STRATEGY", "rule_based"),
        }
    )

    client = get_imap_client()
    processed_count = 0
    lead_count = 0
    skipped_count = 0
    
    try:
        uids = search_since_days(client, folder, since_days)
        logger.info(
            "Found messages to process",
            extra={
                "folder": folder,
                "message_count": len(uids),
                "since_days": since_days,
            }
        )

        for uid in uids:
            msg = fetch_message(client, uid)
            headers = extract_headers(msg)
            msg_id = (headers.get("Message-ID") or "").strip()
            uid_str = str(uid)
            from_domain = extract_sender_domain(headers)
            log_context = {
                "imap_uid": uid_str,
                "message_id": msg_id or uid_str,
                "from_domain": from_domain,
            }

            with SessionLocal() as db:
                if _already_processed(db, msg_id, uid_str):
                    logger.debug(
                        "Email already processed",
                        extra={**log_context, "esito": "duplicate"},
                    )
                    skipped_count += 1
                    continue

                is_allowed, reason = allowed_sender(headers)
                if not is_allowed:
                    skip_reason = reason or f"Sender not allowed: {headers.get('From')}"
                    _mark_sender_disallowed(
                        db,
                        message_id=msg_id or None,
                        uid_str=uid_str,
                        from_domain=from_domain,
                        reason=skip_reason,
                    )
                    skipped_count += 1
                    continue

                body = get_text_body(msg)
                is_lead, score, confidence = matches_lead_keywords(
                    headers,
                    body,
                    return_details=True,
                )
                
                if not is_lead:
                    _mark_sender_disallowed(
                        db,
                        message_id=msg_id or None,
                        uid_str=uid_str,
                        from_domain=from_domain,
                        reason=f"Missing lead keywords (score={score:.2f}, conf={confidence})",
                    )
                    logger.debug(
                        "Email rejected by lead classifier",
                        extra={
                            **log_context,
                            "esito": "not_lead",
                            "score": score,
                            "confidence": confidence,
                        }
                    )
                    skipped_count += 1
                    continue

                received_at = parse_received_at(headers)
                result = process_incoming_email(
                    db,
                    headers=headers,
                    body=body,
                    imap_uid=uid,
                    received_at=received_at,
                )

                result_context = {**log_context}
                contact_id = result.get("contact_id")
                if contact_id is not None:
                    result_context["contact_id"] = contact_id

                if result["status"] == "processed":
                    processed_count += 1
                    if result.get("created"):
                        lead_count += 1
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
                                notifier.send_new_lead({
                                    **{k: v for k, v in lead_payload.items() if isinstance(v, str)},
                                    "received_at": lead_payload["received_at"].isoformat()
                                    if isinstance(lead_payload.get("received_at"), datetime)
                                    else str(lead_payload.get("received_at") or ""),
                                })
                                if excel_writer and excel_row:
                                    notifier.send_excel_update(
                                        {
                                            **{k: v for k, v in lead_payload.items() if isinstance(v, str)},
                                            "received_at": lead_payload["received_at"].isoformat()
                                            if isinstance(lead_payload.get("received_at"), datetime)
                                            else str(lead_payload.get("received_at") or ""),
                                        },
                                        workbook_path=str(excel_writer.path),
                                        row_number=excel_row,
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
                    logger.info(
                        "Email skipped by ingestion",
                        extra={
                            **result_context, 
                            "esito": "skipped",
                            "reason": result.get("reason", "unknown"),
                        },
                    )
                    skipped_count += 1

    except Exception as exc:
        logger.error(
            "Ingestion failed with exception",
            exc_info=True,
            extra={"esito": "error"}
        )
        raise
    finally:
        with suppress(Exception):
            client.logout()
        
        # Summary metrics
        logger.info(
            "Ingestion completed",
            extra={
                "processed": processed_count,
                "new_leads": lead_count,
                "skipped": skipped_count,
                "total": len(uids),
            }
        )


if __name__ == "__main__":
    main()