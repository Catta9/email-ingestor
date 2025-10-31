from __future__ import annotations

import asyncio
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import AsyncGenerator, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware

from libs.db import SessionLocal, init_db
from libs.lead_storage import build_structured_workbook
from libs.models import Contact
from libs.services.ingestion_runner import IngestionEvent, IngestionRunner
from libs.metrics import CONTENT_TYPE_LATEST, render_metrics


logger = logging.getLogger(__name__)

app = FastAPI(title="Email → CRM/Excel Ingestor")


class MetricsEndpointMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if request.url.path == "/metrics":
            body = render_metrics()
            return Response(content=body, media_type=CONTENT_TYPE_LATEST)
        return await call_next(request)


app.add_middleware(MetricsEndpointMiddleware)


class EventBroadcaster:
    """Manage ingestion runs and publish events to subscribers."""

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
        self._loop = loop

    async def subscribe(self) -> asyncio.Queue[IngestionEvent]:
        queue: asyncio.Queue[IngestionEvent] = asyncio.Queue()
        queue.put_nowait(self._last_status)
        self._queues.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[IngestionEvent]) -> None:
        self._queues.discard(queue)

    async def start_run(self) -> None:
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
        self.publish({"type": "status", "message": message, "data": {"state": state}})

    def publish(self, event: IngestionEvent) -> None:
        if self._loop is None:
            return

        def _broadcast(evt: IngestionEvent) -> None:
            if evt.get("type") == "status":
                self._last_status = evt
            for queue in list(self._queues):
                queue.put_nowait(evt)

        self._loop.call_soon_threadsafe(_broadcast, event)

    def _finish_run(self) -> None:
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
def list_contacts(limit: int = 100, offset: int = 0) -> List[Dict[str, object]]:
    with SessionLocal() as db:
        stmt = select(Contact).limit(limit).offset(offset)
        rows = db.execute(stmt).scalars().all()
        return [
            {
                "id": c.id,
                "email": c.email,
                "first_name": c.first_name,
                "last_name": c.last_name,
                "phone": c.phone,
                "org": c.org,
                "source": c.source,
                "created_at": c.created_at.isoformat(),
                "last_message_subject": c.last_message_subject,
                "last_message_received_at": c.last_message_received_at.isoformat()
                if c.last_message_received_at
                else None,
                "last_message_excerpt": c.last_message_excerpt,
            }
            for c in rows
        ]


@app.post("/ingestion/run")
async def trigger_ingestion() -> Dict[str, str]:
    try:
        await events.start_run()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "started"}


@app.get("/ingestion/stream")
async def ingestion_stream() -> StreamingResponse:
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
def export_xlsx() -> StreamingResponse:
    with SessionLocal() as db:
        rows = db.execute(select(Contact)).scalars().all()

    headers = [
        "id",
        "email",
        "first_name",
        "last_name",
        "phone",
        "org",
        "source",
        "created_at",
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
