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
