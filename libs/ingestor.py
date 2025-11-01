from __future__ import annotations

import logging
from datetime import datetime
from email.utils import parseaddr
from typing import Any, Dict
from html import unescape
from bs4 import BeautifulSoup
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Contact, ContactEvent, ProcessedMessage
from .parser import parse_contact_fields


logger = logging.getLogger(__name__)


def extract_sender_domain(headers: dict[str, str]) -> str | None:
    """Restituisce il dominio del mittente partendo dagli header."""

    raw_from = headers.get("From") or ""
    _, email_address = parseaddr(raw_from)
    email_address = email_address.strip().lower()
    if "@" not in email_address:
        return None
    return email_address.rsplit("@", 1)[1]


class IngestionResult(Dict[str, Any]):
    """Risultato tipizzato restituito da ``process_incoming_email``."""

def _plain_text_body(body: str) -> str:
    """Converte il corpo dell'email in testo semplice ripulito dall'HTML."""

    text = body
    if "<" in body and ">" in body:
        soup = BeautifulSoup(body, "html.parser")
        text = soup.get_text(separator=" ")
    text = unescape(text).replace("\xa0", " ")
    return " ".join(text.split())


def _build_body_excerpt(body: str, max_chars: int = 400) -> str:
    """Restituisce un estratto monolinea del corpo email."""

    normalized = _plain_text_body(body)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1] + "…"


def _is_already_processed(db: Session, *, message_id: str | None, uid: str | None) -> bool:
    """Verifica se il messaggio è già stato elaborato (idempotenza)."""

    if message_id:
        processed = (
            db.execute(select(ProcessedMessage).where(ProcessedMessage.message_id == message_id))
            .scalar_one_or_none()
        )
        if processed is not None:
            return True

    if uid:
        processed = (
            db.execute(select(ProcessedMessage).where(ProcessedMessage.imap_uid == uid))
            .scalar_one_or_none()
        )
        if processed is not None:
            return True

    return False


def process_incoming_email(
    db: Session,
    headers: dict[str, str] | None,
    body: str,
    imap_uid: int | str | None = None,
    *,
    received_at: datetime | None = None,
) -> IngestionResult:
    """Processa un'email e persiste i record necessari nel database."""

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

    # Idempotenza: controlla prima per Message-ID, poi per UID IMAP.
    if _is_already_processed(db, message_id=message_id or None, uid=None):
        logger.info(
            "Email già elaborata (message-id)",
            extra={**context, "esito": "duplicate"},
        )
        return IngestionResult({"status": "skipped", "reason": "already_processed"})

    if _is_already_processed(db, message_id=None, uid=uid_str):
        logger.info(
            "Email già elaborata (imap-uid)",
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
            "Email ignorata: campo email non trovato",
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
        db.flush()  # assicura che l'ID sia disponibile
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
    logger.info(
        "Email ingerita con successo",
        extra={**final_context, "esito": "ingested"},
    )
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
