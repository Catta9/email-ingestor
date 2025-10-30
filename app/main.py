from __future__ import annotations
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from io import BytesIO
from openpyxl import Workbook

from libs.db import SessionLocal, init_db
from libs.models import Contact

app = FastAPI(title="Email → CRM/Excel Ingestor")

@app.on_event("startup")
def startup():
    init_db()

## check rapido dello stato
@app.get("/health")
def health():
    return {"status": "ok"}

## ritorna la lista dei contatti da db
@app.get("/contacts")
def list_contacts(limit: int = 100, offset: int = 0):
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
            } for c in rows
        ]

## genera e ritorna un file xlsx con i contatti
@app.get("/export/xlsx")
def export_xlsx():
    with SessionLocal() as db:
        rows = db.execute(select(Contact)).scalars().all()

    wb = Workbook()
    ws = wb.active
    ws.title = "Contacts"
    headers = ["id", "email", "first_name", "last_name", "phone", "org", "source", "created_at"]
    ws.append(headers)
    for c in rows:
        ws.append([
            c.id, c.email, c.first_name, c.last_name, c.phone, c.org, c.source,
            c.created_at.isoformat() if c.created_at else None
        ])

    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="contacts_export.xlsx"'}
    )
