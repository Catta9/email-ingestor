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


def test_lead_scorer_detects_linguistic_variants(monkeypatch):
    monkeypatch.delenv("LEAD_KEYWORDS", raising=False)
    monkeypatch.setenv("LEAD_SCORE_THRESHOLD", "2.0")
    monkeypatch.delenv("LEAD_NEGATIVE_KEYWORDS", raising=False)
    run_ingestor._LEAD_SCORER = None

    headers = {"Subject": "Richiesta preventivi"}
    assert run_ingestor.matches_lead_keywords(headers, "") is True

    run_ingestor._LEAD_SCORER = None
    headers = {"Subject": "Quote request for services"}
    assert run_ingestor.matches_lead_keywords(headers, "") is True

    run_ingestor._LEAD_SCORER = None
    headers = {"Subject": "Richiesta informazioni"}
    body = "Serve una quotazione urgente"
    assert run_ingestor.matches_lead_keywords(headers, body) is False


def test_lead_scorer_handles_negative_keywords(monkeypatch):
    monkeypatch.delenv("LEAD_KEYWORDS", raising=False)
    monkeypatch.setenv("LEAD_SCORE_THRESHOLD", "2.0")
    monkeypatch.setenv("LEAD_NEGATIVE_KEYWORDS", "non serve preventivo")
    run_ingestor._LEAD_SCORER = None

    headers = {"Subject": "Richiesta preventivo"}
    body = "In realta non serve preventivo al momento"
    assert run_ingestor.matches_lead_keywords(headers, body) is False


def test_lead_scorer_combines_subject_and_body(monkeypatch):
    monkeypatch.delenv("LEAD_KEYWORDS", raising=False)
    monkeypatch.setenv("LEAD_SCORE_THRESHOLD", "2.0")
    monkeypatch.delenv("LEAD_NEGATIVE_KEYWORDS", raising=False)
    run_ingestor._LEAD_SCORER = None

    headers = {"Subject": "Richiesta informazioni"}
    body = "Vorremmo una quotazione e un'offerta dettagliata"
    assert run_ingestor.matches_lead_keywords(headers, body) is True

    run_ingestor._LEAD_SCORER = None
    body = "Vorremmo solo una quotazione"
    assert run_ingestor.matches_lead_keywords(headers, body) is False
