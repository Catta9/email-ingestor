from __future__ import annotations
from typing import Any, Dict

from sqlalchemy import select

from .models import Contact, ContactEvent, ProcessedMessage
from .parser import parse_contact_fields


class IngestionResult(Dict[str, Any]):
    """Typed dict-like result for process_incoming_email."""


def process_incoming_email(db, headers: dict[str, str] | None, body: str, imap_uid: int | str | None = None) -> IngestionResult:
    """Process a single email payload and persist relevant records.

    The function handles idempotency using the Message-ID or IMAP UID. It
    returns a dictionary describing whether the message has been processed
    or skipped (and why).
    """
    headers = headers or {}
    message_id = (headers.get("Message-ID") or "").strip()
    uid_str = str(imap_uid) if imap_uid is not None else None

    # Idempotency: prefer Message-ID, fallback to IMAP UID.
    if message_id:
        processed = db.execute(
            select(ProcessedMessage).where(ProcessedMessage.message_id == message_id)
        ).scalar_one_or_none()
        if processed is not None:
            return IngestionResult({"status": "skipped", "reason": "already_processed"})

    if uid_str:
        processed = db.execute(
            select(ProcessedMessage).where(ProcessedMessage.imap_uid == uid_str)
        ).scalar_one_or_none()
        if processed is not None:
            return IngestionResult({"status": "skipped", "reason": "already_processed"})

    fields = parse_contact_fields(body, headers=headers)
    email_val = fields.get("email")
    fallback_message_id = message_id or (f"uid:{uid_str}" if uid_str else None)

    if not email_val:
        if fallback_message_id:
            db.add(ProcessedMessage(message_id=fallback_message_id, imap_uid=uid_str))
            db.commit()
        return IngestionResult({"status": "skipped", "reason": "missing_email"})

    existing = db.execute(select(Contact).where(Contact.email == email_val)).scalar_one_or_none()

    if existing:
        contact = existing
        changed = False
        for key in ("first_name", "last_name", "phone", "org"):
            value = fields.get(key)
            if value and not getattr(contact, key):
                setattr(contact, key, value)
                changed = True
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
        )
        db.add(contact)
        db.flush()  # populate primary key

    event = ContactEvent(
        contact_id=contact.id,
        event_type="email_inbound",
        payload={
            "headers": headers,
            "extracted": fields,
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
    return IngestionResult({"status": "processed", "contact_id": contact.id})
