from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo


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

DATA_SHEET_TITLE = "Leads"
SUMMARY_SHEET_TITLE = "Summary"
DEFAULT_TABLE_NAME = "LeadsTable"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _coerce_datetime(value: object) -> object:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return value
    return value


def _header_index(headers: Sequence[str], target: str) -> Optional[int]:
    lookup = {name.lower(): idx for idx, name in enumerate(headers, start=1)}
    return lookup.get(target.lower())


def _apply_header_style(ws, header_count: int) -> None:
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="305496", end_color="305496", fill_type="solid")
    alignment = Alignment(horizontal="center", vertical="center")
    for col_idx in range(1, header_count + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = alignment


def _auto_fit_columns(ws) -> None:
    for column_cells in ws.columns:
        column_cells = list(column_cells)
        if not column_cells:
            continue
        column_letter = get_column_letter(column_cells[0].column)
        max_length = 0
        for cell in column_cells:
            value = cell.value
            length = len(str(value)) if value is not None else 0
            max_length = max(max_length, length)
        ws.column_dimensions[column_letter].width = min(max_length + 2, 60)


def _ensure_table(ws, header_count: int, *, table_name: str) -> None:
    max_row = ws.max_row or 1
    max_col = ws.max_column or header_count
    ref = f"A1:{get_column_letter(max_col)}{max_row}"
    if table_name in ws.tables:
        ws.tables[table_name].ref = ref
        return

    table = Table(displayName=table_name, ref=ref)
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium9",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    ws.add_table(table)


def _email_domain(value: object) -> Optional[str]:
    if not value:
        return None
    email = str(value).strip().lower()
    if "@" not in email:
        return None
    return email.rsplit("@", 1)[1]


def _collect_column_values(ws, column_index: int) -> list[object]:
    values = []
    for row_idx in range(2, ws.max_row + 1):
        value = ws.cell(row=row_idx, column=column_index).value
        if value:
            values.append(value)
    return values


def _style_header_row(ws, row_idx: int, column_count: int) -> None:
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    alignment = Alignment(horizontal="center")
    for col in range(1, column_count + 1):
        cell = ws.cell(row=row_idx, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = alignment


def _update_summary_sheet(
    wb: Workbook,
    headers: Sequence[str],
    *,
    data_sheet_name: str = DATA_SHEET_TITLE,
    summary_sheet_name: str = SUMMARY_SHEET_TITLE,
) -> None:
    header_list = list(headers)
    email_idx = _header_index(header_list, "email")
    company_idx = _header_index(header_list, "company") or _header_index(header_list, "org")

    if summary_sheet_name in wb.sheetnames:
        summary_ws = wb[summary_sheet_name]
        summary_ws.delete_rows(1, summary_ws.max_row)
    else:
        summary_ws = wb.create_sheet(summary_sheet_name, 0)

    data_ws = wb[data_sheet_name]
    total_leads = max(data_ws.max_row - 1, 0)
    summary_ws.append(["Metric", "Value"])
    _style_header_row(summary_ws, 1, 2)
    summary_ws.freeze_panes = "A2"
    summary_ws.column_dimensions["A"].width = 28
    summary_ws.column_dimensions["B"].width = 40

    summary_ws.append(["Totale lead", total_leads])

    if company_idx:
        companies = {
            str(value).strip().lower()
            for value in _collect_column_values(data_ws, company_idx)
            if str(value).strip()
        }
        summary_ws.append(["Aziende uniche", len(companies)])

    if email_idx:
        domain_counter: Counter[str] = Counter()
        for value in _collect_column_values(data_ws, email_idx):
            domain = _email_domain(value)
            if domain:
                domain_counter[domain] += 1

        if domain_counter:
            summary_ws.append([])
            summary_ws.append(["Dominio email", "Lead"])
            header_row = summary_ws.max_row
            _style_header_row(summary_ws, header_row, 2)
            for domain, count in domain_counter.most_common(10):
                summary_ws.append([domain, count])

    summary_ws.append([])
    summary_ws.append(["Ultimo aggiornamento", datetime.utcnow()])

    _auto_fit_columns(summary_ws)

    idx = wb.sheetnames.index(summary_sheet_name)
    if idx != 0:
        wb.move_sheet(summary_ws, -idx)

    wb.active = wb.sheetnames.index(data_sheet_name)


def _finalize_data_sheet(ws, header_count: int, table_name: str) -> None:
    ws.freeze_panes = "A2"
    _apply_header_style(ws, header_count)
    _ensure_table(ws, header_count, table_name=table_name)
    _auto_fit_columns(ws)


def build_structured_workbook(
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    data_sheet_name: str = DATA_SHEET_TITLE,
    summary_sheet_name: str = SUMMARY_SHEET_TITLE,
    table_name: str = DEFAULT_TABLE_NAME,
) -> Workbook:
    wb = Workbook()
    data_ws = wb.active
    data_ws.title = data_sheet_name
    header_list = list(headers)
    data_ws.append(header_list)
    for row in rows:
        data_ws.append(list(row))

    _finalize_data_sheet(data_ws, len(header_list), table_name)
    _update_summary_sheet(
        wb,
        header_list,
        data_sheet_name=data_sheet_name,
        summary_sheet_name=summary_sheet_name,
    )
    return wb


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

    def append(self, lead: dict[str, Optional[str | datetime]]) -> int:
        headers = list(self.headers)
        _ensure_parent(self.path)
        if self.path.exists():
            wb = load_workbook(self.path)
            data_ws = wb[DATA_SHEET_TITLE] if DATA_SHEET_TITLE in wb.sheetnames else wb.active
        else:
            wb = Workbook()
            data_ws = wb.active
            data_ws.title = DATA_SHEET_TITLE
            data_ws.append(headers)

        row = [
            _coerce_datetime(lead.get("inserted_at")),
            lead.get("email"),
            lead.get("first_name"),
            lead.get("last_name"),
            lead.get("company"),
            lead.get("phone"),
            lead.get("subject"),
            _coerce_datetime(lead.get("received_at")),
            lead.get("notes"),
        ]

        data_ws.append(row)
        _finalize_data_sheet(data_ws, len(headers), DEFAULT_TABLE_NAME)
        _update_summary_sheet(wb, headers)
        wb.save(self.path)
        return data_ws.max_row
