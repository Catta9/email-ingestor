from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Text, JSON, DateTime, Boolean

from .db import Base

def gen_uuid() -> str:
    return str(uuid.uuid4())

## definizione dei modelli ORM (= Object-Relational Mapping)
## In pratica: invece di scrivere SQL a mano, definisci classi Python che rappresentano le tabelle. Ogni istanza della classe = una riga nel DB.
## record "pulito" dei contatti
class Contact(Base):
    __tablename__ = "contacts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    email: Mapped[str] = mapped_column(String, nullable=False, index=True)
    first_name: Mapped[str | None] = mapped_column(String, nullable=True)
    last_name: Mapped[str | None] = mapped_column(String, nullable=True)
    phone: Mapped[str | None] = mapped_column(String, nullable=True)
    org: Mapped[str | None] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, default="email")
    consent: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_message_subject: Mapped[str | None] = mapped_column(String, nullable=True)
    last_message_received_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_message_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)

## audit/eventi -> traccia provenienza e payload estratto 
class ContactEvent(Base):
    __tablename__ = "contact_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    contact_id: Mapped[str] = mapped_column(String, index=True)
    event_type: Mapped[str] = mapped_column(String)  # email_inbound|form_submit|updated
    payload: Mapped[dict] = mapped_column(JSON)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

## serve all'idempotenza: memorizza message-id delle email già processate, cos' non le rielabora
class ProcessedMessage(Base):
    __tablename__ = "processed_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    message_id: Mapped[str] = mapped_column(String, unique=True, index=True)  # RFC822 Message-ID
    imap_uid: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    processed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
