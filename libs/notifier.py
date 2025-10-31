from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from datetime import datetime
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

    def _send_message(self, subject: str, lines: Sequence[str]) -> bool:
        if not self.recipients:
            return False

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)
        msg.set_content("\n".join(lines))

        with smtplib.SMTP(self.host, self.port, timeout=20) as smtp:
            if self.use_tls:
                smtp.starttls()
            if self.username and self.password:
                smtp.login(self.username, self.password)
            smtp.send_message(msg)
        return True

    def send_new_lead(self, lead: dict[str, str]) -> bool:
        subject = lead.get("subject") or "Nuovo lead ricevuto"
        company = lead.get("org") or lead.get("company") or ""
        lines = [
            "Nuovo lead inserito automaticamente:",
            f"Nome: {lead.get('first_name') or ''} {lead.get('last_name') or ''}",
            f"Email: {lead.get('email') or ''}",
            f"Azienda: {company}",
            f"Telefono: {lead.get('phone') or ''}",
            f"Ricevuto alle: {lead.get('received_at') or ''}",
            "",
            lead.get("notes") or "",
        ]
        return self._send_message(subject, lines)

    def send_excel_update(
        self,
        lead: dict[str, str],
        *,
        workbook_path: str,
        row_number: int | None = None,
    ) -> bool:
        subject = "AVVISO: è stato aggiunto un cliente al foglio excel!"
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        company = lead.get("org") or lead.get("company") or ""
        lines = [
            subject,
            "",
            f"Email cliente: {lead.get('email') or ''}",
            f"Nome: {lead.get('first_name') or ''} {lead.get('last_name') or ''}",
            f"Azienda: {company}",
            f"Telefono: {lead.get('phone') or ''}",
            f"File Excel: {workbook_path}",
        ]
        if row_number:
            lines.append(f"Riga inserita: {row_number}")
        lines.extend([
            f"Aggiornato alle: {timestamp}",
            "",
            "Estratto nota:",
            lead.get("notes") or "",
        ])
        return self._send_message(subject, lines)
