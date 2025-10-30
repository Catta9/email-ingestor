from __future__ import annotations

from sqlalchemy import select

from libs.ingestor import process_incoming_email
from libs.models import Contact, ContactEvent, ProcessedMessage


def _make_headers(message_id: str = "<msg-1@example.com>") -> dict[str, str]:
    return {
        "Message-ID": message_id,
        "From": "Jane Doe <jane.doe@example.com>",
        "To": "ingest@example.com",
        "Subject": "New lead",
    }


def test_process_incoming_email_creates_contact(session):
    body = """Hello, this is Jane.
    Phone: +39 333 1234567
    Company: Example Inc
    """
    result = process_incoming_email(session, headers=_make_headers(), body=body, imap_uid=101)

    assert result["status"] == "processed"
    contact_id = result["contact_id"]

    contact = session.execute(select(Contact).where(Contact.id == contact_id)).scalar_one()
    assert contact.email == "jane.doe@example.com"
    assert contact.first_name == "Jane"
    assert contact.last_name == "Doe"
    assert contact.phone.endswith("1234567")
    assert contact.org == "Example Inc"

    events = session.execute(select(ContactEvent)).scalars().all()
    assert len(events) == 1
    processed = session.execute(select(ProcessedMessage)).scalars().all()
    assert len(processed) == 1


def test_process_incoming_email_is_idempotent(session):
    body = """Hello, this is Jane.
    Phone: +39 333 1234567
    Company: Example Inc
    """
    headers = _make_headers()

    first = process_incoming_email(session, headers=headers, body=body, imap_uid=202)
    assert first["status"] == "processed"

    second = process_incoming_email(session, headers=headers, body=body, imap_uid=202)
    assert second["status"] == "skipped"
    assert second["reason"] == "already_processed"

    contacts = session.execute(select(Contact)).scalars().all()
    assert len(contacts) == 1
    events = session.execute(select(ContactEvent)).scalars().all()
    assert len(events) == 1
    processed_entries = session.execute(select(ProcessedMessage)).scalars().all()
    assert len(processed_entries) == 1


def test_process_incoming_email_idempotent_with_missing_message_id(session):
    body = """Hello team,
    Phone: 0123456789
    """
    headers = _make_headers(message_id="")

    first = process_incoming_email(session, headers=headers, body=body, imap_uid=303)
    assert first["status"] == "processed"

    second = process_incoming_email(session, headers=headers, body=body, imap_uid=303)
    assert second["status"] == "skipped"
    assert second["reason"] == "already_processed"

    processed_entries = session.execute(select(ProcessedMessage)).scalars().all()
    assert len(processed_entries) == 1
