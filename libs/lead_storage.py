from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from openpyxl import Workbook, load_workbook


DEFAULT_HEADERS: tuple[str, ...] = (
    "inserted_at",
    "email",
    "first_name",
    "last_name",
    "company",
    "phone",
    "subject",
    "received_at",
    "notes",
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat(timespec="seconds")


@dataclass
class ExcelLeadWriter:
    """Append lead rows to an Excel workbook, creating it if needed."""

    path: Path
    headers: Iterable[str] = DEFAULT_HEADERS

    @classmethod
    def from_env(cls) -> "ExcelLeadWriter | None":
        target = os.getenv("LEADS_XLSX_PATH", "data/leads.xlsx").strip()
        if not target:
            return None
        return cls(Path(target))

    def append(self, lead: dict[str, Optional[str | datetime]]) -> None:
        _ensure_parent(self.path)
        if self.path.exists():
            wb = load_workbook(self.path)
            ws = wb.active
        else:
            wb = Workbook()
            ws = wb.active
            ws.title = "Leads"
            ws.append(list(self.headers))

        row = [
            _format_datetime(lead.get("inserted_at")) if isinstance(lead.get("inserted_at"), datetime) else lead.get("inserted_at"),
            lead.get("email"),
            lead.get("first_name"),
            lead.get("last_name"),
            lead.get("company"),
            lead.get("phone"),
            lead.get("subject"),
            _format_datetime(lead.get("received_at")) if isinstance(lead.get("received_at"), datetime) else lead.get("received_at"),
            lead.get("notes"),
        ]

        ws.append(row)
        wb.save(self.path)
