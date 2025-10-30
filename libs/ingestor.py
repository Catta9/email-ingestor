from __future__ import annotations
import logging
from email.utils import parseaddr
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import select

from .models import Contact, ContactEvent, ProcessedMessage
from .parser import parse_contact_fields


logger = logging.getLogger(__name__)


def extract_sender_domain(headers: dict[str, str]) -> str | None:
    """Return the sender domain from the headers if present."""

    raw_from = headers.get("From") or ""
    _, email_address = parseaddr(raw_from)
    email_address = email_address.strip().lower()
    if "@" not in email_address:
        return None
    return email_address.rsplit("@", 1)[1]


class IngestionResult(Dict[str, Any]):
    """Typed dict-like result for process_incoming_email."""


def _build_body_excerpt(body: str, max_chars: int = 400) -> str:
    """Return a compact single-line excerpt of the body."""

    normalized = " ".join(body.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1] + "…"


def process_incoming_email(
    db,
    headers: dict[str, str] | None,
    body: str,
    imap_uid: int | str | None = None,
    *,
    received_at: datetime | None = None,
) -> IngestionResult:
    """Process a single email payload and persist relevant records.

    The function handles idempotency using the Message-ID or IMAP UID. It
    returns a dictionary describing whether the message has been processed
    or skipped (and why).
    """
    headers = headers or {}
    message_id = (headers.get("Message-ID") or "").strip()
    uid_str = str(imap_uid) if imap_uid is not None else None

    from_domain = extract_sender_domain(headers)
    identifier = message_id or uid_str

    context: dict[str, str | int | None] = {
        "imap_uid": uid_str,
        "message_id": identifier,
        "from_domain": from_domain,
    }

    # Idempotency: prefer Message-ID, fallback to IMAP UID.
    if message_id:
        processed = db.execute(
            select(ProcessedMessage).where(ProcessedMessage.message_id == message_id)
        ).scalar_one_or_none()
        if processed is not None:
            logger.info(
                "Email already processed (message-id)",
                extra={**context, "esito": "duplicate"},
            )
            return IngestionResult({"status": "skipped", "reason": "already_processed"})

    if uid_str:
        processed = db.execute(
            select(ProcessedMessage).where(ProcessedMessage.imap_uid == uid_str)
        ).scalar_one_or_none()
        if processed is not None:
            logger.info(
                "Email already processed (imap-uid)",
                extra={**context, "esito": "duplicate"},
            )
            return IngestionResult({"status": "skipped", "reason": "already_processed"})

    fields = parse_contact_fields(body, headers=headers)
    email_val = fields.get("email")
    fallback_message_id = message_id or (f"uid:{uid_str}" if uid_str else None)
    if fallback_message_id and context["message_id"] is None:
        context["message_id"] = fallback_message_id

    if not email_val:
        if fallback_message_id:
            db.add(ProcessedMessage(message_id=fallback_message_id, imap_uid=uid_str))
            db.commit()
        logger.warning(
            "Email skipped due to missing email field",
            extra={**context, "esito": "skipped"},
        )
        return IngestionResult({"status": "skipped", "reason": "missing_email"})

    existing = db.execute(select(Contact).where(Contact.email == email_val)).scalar_one_or_none()

    subject = headers.get("Subject") if headers else None
    excerpt = _build_body_excerpt(body)

    if existing:
        contact = existing
        changed = False
        for key in ("first_name", "last_name", "phone", "org"):
            value = fields.get(key)
            if value and not getattr(contact, key):
                setattr(contact, key, value)
                changed = True
        contact.last_message_subject = subject or contact.last_message_subject
        contact.last_message_received_at = received_at or contact.last_message_received_at
        contact.last_message_excerpt = excerpt
        is_new_contact = False
        if changed:
            db.add(contact)
    else:
        contact = Contact(
            email=email_val,
            first_name=fields.get("first_name"),
            last_name=fields.get("last_name"),
            phone=fields.get("phone"),
            org=fields.get("org"),
            source="email",
            last_message_subject=subject,
            last_message_received_at=received_at,
            last_message_excerpt=excerpt,
        )
        db.add(contact)
        db.flush()  # populate primary key
        is_new_contact = True

    event = ContactEvent(
        contact_id=contact.id,
        event_type="email_inbound",
        payload={
            "headers": headers,
            "extracted": fields,
            "received_at": received_at.isoformat() if received_at else None,
            "body_excerpt": excerpt,
        },
    )
    db.add(event)

    db.add(
        ProcessedMessage(
            message_id=fallback_message_id or contact.id,
            imap_uid=uid_str,
        )
    )

    db.commit()
    final_context = {**context, "contact_id": contact.id}
    if final_context.get("message_id") is None:
        final_context["message_id"] = str(contact.id)
    logger.info("Email ingested successfully", extra={**final_context, "esito": "ingested"})
    return IngestionResult(
        {
            "status": "processed",
            "contact_id": contact.id,
            "created": is_new_contact,
            "extracted": fields,
            "subject": subject,
            "received_at": received_at,
            "body_excerpt": excerpt,
        }
    )
