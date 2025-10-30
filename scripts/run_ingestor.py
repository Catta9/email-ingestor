from __future__ import annotations

import logging
import os
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


logger = logging.getLogger(__name__)

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


def matches_lead_keywords(headers: dict[str, str], body: str) -> bool:
    classifier = _get_ml_classifier() if _use_ml_strategy() else None
    if classifier is not None:
        if classifier.is_relevant(headers, body):
            return True
        if _use_hybrid_strategy():
            scorer = _get_rule_based_scorer()
            return scorer.is_relevant(headers, body)
        return False

    scorer = _get_rule_based_scorer()
    return scorer.is_relevant(headers, body)


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

## è lo script che userà prod dev
def main():
    ## assicura che ci siano le tabelle
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
    init_db()
    folder = os.getenv("IMAP_FOLDER", "INBOX")
    since_days = int(os.getenv("IMAP_SEARCH_SINCE_DAYS", "7"))

    excel_writer = ExcelLeadWriter.from_env()
    notifier = EmailNotifier.from_env()

    client = get_imap_client()
    ## apre IMAP e cerca i messaggi recenti
    try:
        uids = search_since_days(client, folder, since_days)
        logger.info(
            "Found %d messages since %d days",
            len(uids),
            since_days,
            extra={"folder": folder},
        )

        ## per ogni UID: estrae header e Message-ID -> controlla idempotenza ->estrae corpo e lo manda al parser
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
                    logger.info(
                        "Email already processed (runner pre-check)",
                        extra={**log_context, "esito": "duplicate"},
                    )
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
                    continue

                body = get_text_body(msg)
                if not matches_lead_keywords(headers, body):
                    _mark_sender_disallowed(
                        db,
                        message_id=msg_id or None,
                        uid_str=uid_str,
                        from_domain=from_domain,
                        reason="Missing lead keywords",
                    )
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
                    if result.get("created"):
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
                        if excel_writer:
                            excel_writer.append(lead_payload)
                        if notifier:
                            with suppress(Exception):
                                notifier.send_new_lead({
                                    **{k: v for k, v in lead_payload.items() if isinstance(v, str)},
                                    "received_at": lead_payload["received_at"].isoformat()
                                    if isinstance(lead_payload.get("received_at"), datetime)
                                    else str(lead_payload.get("received_at") or ""),
                                })
                    logger.info(
                        "Email processed",
                        extra={**result_context, "esito": "ingested"},
                    )
                else:
                    logger.info(
                        "Email skipped by ingestion: %s",
                        result.get("reason", "unknown"),
                        extra={**result_context, "esito": "skipped"},
                    )

    finally:
        with suppress(Exception):
            client.logout()

if __name__ == "__main__":
    main()
