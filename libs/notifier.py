from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Sequence


def _parse_recipients(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class EmailNotifier:
    """Send email notifications when new leads are ingested."""

    host: str
    port: int
    sender: str
    recipients: Sequence[str]
    username: str | None = None
    password: str | None = None
    use_tls: bool = True

    @classmethod
    def from_env(cls) -> "EmailNotifier | None":
        host = os.getenv("SMTP_HOST")
        sender = os.getenv("SMTP_SENDER")
        recipients = _parse_recipients(os.getenv("NOTIFY_RECIPIENTS"))
        if not (host and sender and recipients):
            return None

        port = int(os.getenv("SMTP_PORT", "587"))
        username = os.getenv("SMTP_USERNAME")
        password = os.getenv("SMTP_PASSWORD")
        use_tls = os.getenv("SMTP_USE_TLS", "true").lower() not in {"0", "false", "no"}
        return cls(
            host=host,
            port=port,
            sender=sender,
            recipients=recipients,
            username=username,
            password=password,
            use_tls=use_tls,
        )

    def send_new_lead(self, lead: dict[str, str]) -> bool:
        if not self.recipients:
            return False

        msg = EmailMessage()
        subject = lead.get("subject") or "Nuovo lead ricevuto"
        msg["Subject"] = subject
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)

        company = lead.get("org") or lead.get("company") or ""
        lines = [
            "Nuovo lead inserito automaticamente:",
            f"Nome: {lead.get('first_name', '')} {lead.get('last_name', '')}",
            f"Email: {lead.get('email', '')}",
            f"Azienda: {company}",
            f"Telefono: {lead.get('phone', '')}",
            f"Ricevuto alle: {lead.get('received_at', '')}",
            "",
            lead.get("notes", ""),
        ]
        msg.set_content("\n".join(lines))

        with smtplib.SMTP(self.host, self.port, timeout=20) as smtp:
            if self.use_tls:
                smtp.starttls()
            if self.username and self.password:
                smtp.login(self.username, self.password)
            smtp.send_message(msg)
        return True
