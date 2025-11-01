"""Applicazione FastAPI che espone dashboard e API per l'ingestione email."""

from __future__ import annotations

import asyncio
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload
from starlette.middleware.base import BaseHTTPMiddleware

from dotenv import load_dotenv
from libs.db import SessionLocal, init_db
from libs.lead_storage import build_structured_workbook
from libs.models import Contact, ContactTag
from libs.services.ingestion_runner import IngestionEvent, IngestionRunner
from libs.metrics import CONTENT_TYPE_LATEST, render_metrics


load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="Email → CRM/Excel Ingestor")


LeadState = Literal["new", "reviewed"]


class ContactUpdate(BaseModel):
    """Payload ammesso per aggiornare lo stato o le note di un contatto."""

    status: LeadState | None = Field(default=None)
    notes: str | None = Field(default=None, max_length=2000)


class TagCreate(BaseModel):
    """Payload per aggiungere un nuovo tag al contatto."""

    tag: str = Field(min_length=1, max_length=50)


def serialize_contact(contact: Contact) -> Dict[str, object]:
    """Trasforma il modello ORM in un dizionario serializzabile JSON."""

    return {
        "id": contact.id,
        "email": contact.email,
        "first_name": contact.first_name,
        "last_name": contact.last_name,
        "phone": contact.phone,
        "org": contact.org,
        "source": contact.source,
        "created_at": contact.created_at.isoformat(),
        "last_message_subject": contact.last_message_subject,
        "last_message_received_at": contact.last_message_received_at.isoformat()
        if contact.last_message_received_at
        else None,
        "last_message_excerpt": contact.last_message_excerpt,
        "status": contact.status,
        "notes": contact.notes,
        "tags": [tag.tag for tag in contact.tags],
    }
class MetricsEndpointMiddleware(BaseHTTPMiddleware):
    """Espone `/metrics` senza coinvolgere il router principale."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if request.url.path == "/metrics":
            body = render_metrics()
            return Response(content=body, media_type=CONTENT_TYPE_LATEST)
        return await call_next(request)


app.add_middleware(MetricsEndpointMiddleware)


class EventBroadcaster:
    """Gestisce l'esecuzione dell'ingestione e i client SSE collegati."""

    def __init__(self) -> None:
        self._queues: set[asyncio.Queue[IngestionEvent]] = set()
        self._lock = asyncio.Lock()
        self._running = False
        self._current_future: asyncio.Future | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner = IngestionRunner()
        self._last_status: IngestionEvent = {
            "type": "status",
            "message": "In attesa",
            "data": {"state": "idle"},
        }

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Imposta l'event loop usato per le notifiche asincrone."""

        self._loop = loop

    async def subscribe(self) -> asyncio.Queue[IngestionEvent]:
        """Registra un nuovo client SSE restituendo la sua coda di eventi."""

        queue: asyncio.Queue[IngestionEvent] = asyncio.Queue()
        queue.put_nowait(self._last_status)
        self._queues.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[IngestionEvent]) -> None:
        """Rimuove il client SSE dalla lista dei sottoscrittori."""

        self._queues.discard(queue)

    async def start_run(self) -> None:
        """Avvia l'ingestione in un thread dedicato, evitando run concorrenti."""

        async with self._lock:
            if self._running:
                raise RuntimeError("Ingestion already running")
            self._running = True

        loop = asyncio.get_running_loop()

        def runner_callback(event: IngestionEvent) -> None:
            self.publish(event)

        def _run() -> None:
            try:
                self._runner.run(event_callback=runner_callback)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Unhandled ingestion error")
            finally:
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(self._finish_run)

        self.publish_status("running", "Ingestione in corso")
        self._current_future = loop.run_in_executor(None, _run)

    def publish_status(self, state: str, message: str) -> None:
        """Pubblica uno stato sintetico (idle/running/error)."""

        self.publish({"type": "status", "message": message, "data": {"state": state}})

    def publish(self, event: IngestionEvent) -> None:
        """Invia un evento a tutte le code registrate."""

        if self._loop is None:
            return

        def _broadcast(evt: IngestionEvent) -> None:
            if evt.get("type") == "status":
                self._last_status = evt
            for queue in list(self._queues):
                queue.put_nowait(evt)

        self._loop.call_soon_threadsafe(_broadcast, event)

    def _finish_run(self) -> None:
        """Ripristina lo stato interno al termine dell'esecuzione."""

        self._running = False
        self._current_future = None
        self.publish_status("idle", "In attesa di una nuova esecuzione")


events = EventBroadcaster()


static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
async def startup() -> None:
    init_db()
    loop = asyncio.get_running_loop()
    events.set_loop(loop)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    index_file = static_dir / "index.html"
    return FileResponse(index_file)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/contacts")
async def list_contacts(
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, object]]:
    with SessionLocal() as db:
        stmt = (
            select(Contact)
            .options(selectinload(Contact.tags))
            .limit(limit)
            .offset(offset)
        )
        rows = db.execute(stmt).scalars().all()
        return [serialize_contact(contact) for contact in rows]


def _get_contact(session: Session, contact_id: str) -> Contact:
    contact = session.get(Contact, contact_id)
    if contact is None:
        raise HTTPException(status_code=404, detail="Contact not found")
    return contact


@app.patch("/contacts/{contact_id}")
async def update_contact(
    contact_id: str,
    payload: ContactUpdate,
) -> Dict[str, object]:
    with SessionLocal() as db:
        contact = _get_contact(db, contact_id)
        if payload.status is not None:
            contact.status = payload.status
        if payload.notes is not None:
            notes = payload.notes.strip()
            contact.notes = notes or None
        db.add(contact)
        db.commit()
        db.refresh(contact)
        return serialize_contact(contact)


@app.post("/contacts/{contact_id}/tags")
async def add_contact_tag(
    contact_id: str,
    payload: TagCreate,
) -> Dict[str, object]:
    tag_value = payload.tag.strip()
    if not tag_value:
        raise HTTPException(status_code=400, detail="Tag cannot be empty")

    with SessionLocal() as db:
        contact = _get_contact(db, contact_id)
        normalized = tag_value
        existing = {tag.tag.lower() for tag in contact.tags}
        if normalized.lower() not in existing:
            contact.tags.append(ContactTag(tag=normalized))
            db.add(contact)
            db.commit()
            db.refresh(contact)
        return serialize_contact(contact)


@app.post("/ingestion/run")
async def trigger_ingestion() -> Dict[str, str]:
    """Avvia l'ingestione manualmente restituendo lo stato avvio."""

    try:
        await events.start_run()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "started"}


@app.get("/ingestion/stream")
async def ingestion_stream() -> StreamingResponse:
    """Espone il flusso SSE con gli eventi dell'ingestione."""

    queue = await events.subscribe()

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            await events.unsubscribe(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/export/xlsx")
async def export_xlsx() -> StreamingResponse:
    """Genera e restituisce l'export Excel aggiornato del database."""

    with SessionLocal() as db:
        rows = (
            db.execute(select(Contact).options(selectinload(Contact.tags)))
            .scalars()
            .all()
        )

    headers = [
        "id",
        "email",
        "first_name",
        "last_name",
        "phone",
        "org",
        "source",
        "created_at",
        "status",
        "tags",
        "notes",
        "last_message_subject",
        "last_message_received_at",
        "last_message_excerpt",
    ]

    data_rows = [
        [
            c.id,
            c.email,
            c.first_name,
            c.last_name,
            c.phone,
            c.org,
            c.source,
            c.created_at,
            c.status,
            ", ".join(tag.tag for tag in c.tags),
            c.notes,
            c.last_message_subject,
            c.last_message_received_at,
            c.last_message_excerpt,
        ]
        for c in rows
    ]

    wb = build_structured_workbook(
        headers,
        data_rows,
        data_sheet_name="Contacts",
        summary_sheet_name="Dashboard",
        table_name="ContactsTable",
    )

    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="contacts_export.xlsx"'},
    )
