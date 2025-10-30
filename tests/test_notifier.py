from __future__ import annotations

import smtplib
from email.message import EmailMessage

from libs.notifier import EmailNotifier


class DummySMTP:
    """Simple SMTP stub that records sent messages."""

    def __init__(self, host, port, timeout):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.started_tls = False
        self.login_args = None
        self.messages: list[EmailMessage] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def starttls(self):
        self.started_tls = True

    def login(self, username, password):
        self.login_args = (username, password)

    def send_message(self, message: EmailMessage):
        self.messages.append(message)


def test_send_new_lead_prefers_org_over_company(monkeypatch):
    sent_messages: list[EmailMessage] = []

    def fake_smtp(host, port, timeout):
        smtp = DummySMTP(host, port, timeout)
        sent_messages.append(smtp)
        return smtp

    monkeypatch.setattr(smtplib, "SMTP", fake_smtp)

    notifier = EmailNotifier(
        host="smtp.example.com",
        port=587,
        sender="noreply@example.com",
        recipients=["ops@example.com"],
        use_tls=False,
    )

    notifier.send_new_lead(
        {
            "first_name": "Mario",
            "last_name": "Rossi",
            "email": "mario.rossi@example.com",
            "org": "Acme S.p.A.",
            "company": "Vecchia Azienda",
        }
    )

    assert sent_messages, "Expected a message to be sent"
    message = sent_messages[0].messages[0]
    body = message.get_content()
    assert "Azienda: Acme S.p.A." in body
    assert "Vecchia Azienda" not in body
