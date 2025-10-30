import logging

from sqlalchemy import select

from scripts import run_ingestor


def test_allowed_sender_accepts_configured_domain(monkeypatch):
    monkeypatch.setenv("ALLOWED_SENDER_DOMAINS", "example.com")
    is_allowed, reason = run_ingestor.allowed_sender({"From": '"User" <USER@Example.COM>'})
    assert is_allowed is True
    assert reason is None


def test_allowed_sender_rejects_unlisted_domain(monkeypatch):
    monkeypatch.setenv("ALLOWED_SENDER_DOMAINS", "example.com")
    is_allowed, reason = run_ingestor.allowed_sender({"From": "person@another.com"})
    assert is_allowed is False
    assert reason == "Sender domain not allowed: another.com"


def test_allowed_sender_handles_invalid_addresses(monkeypatch):
    monkeypatch.setenv("ALLOWED_SENDER_DOMAINS", "example.com")
    is_allowed, reason = run_ingestor.allowed_sender({"From": "not-an-email"})
    assert is_allowed is False
    assert "Invalid sender address" in reason


def test_runner_logs_domain_skip(session_factory, caplog):
    headers = {
        "From": '"Blocked" <blocked@blocked.com>',
        "Message-ID": "<blocked@example.com>",
    }

    session_cls = session_factory
    db = session_cls()
    try:
        with caplog.at_level(logging.INFO, logger="scripts.run_ingestor"):
            run_ingestor._mark_sender_disallowed(
                db,
                message_id=headers["Message-ID"],
                uid_str="500",
                from_domain="blocked.com",
                reason="Sender domain not allowed: blocked.com",
            )
    finally:
        db.close()

    records = [record for record in caplog.records if getattr(record, "esito", "") == "skipped"]
    assert records, "Expected skip log entry"
    record = records[-1]
    assert record.imap_uid == "500"
    assert record.message_id == "<blocked@example.com>"
    assert record.from_domain == "blocked.com"

    with session_cls() as verify_db:
        stored = verify_db.execute(select(run_ingestor.ProcessedMessage)).scalars().all()
    assert stored, "ProcessedMessage should be recorded"
