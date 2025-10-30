from datetime import datetime
from pathlib import Path

from libs.lead_storage import ExcelLeadWriter


def test_excel_lead_writer_creates_file(tmp_path: Path):
    path = tmp_path / "leads.xlsx"
    writer = ExcelLeadWriter(path)

    lead = {
        "inserted_at": datetime(2024, 9, 24, 15, 45),
        "email": "lead@example.com",
        "first_name": "Lead",
        "last_name": "Example",
        "company": "Example Corp",
        "phone": "+3912345678",
        "subject": "Richiesta preventivo",
        "received_at": datetime(2024, 9, 24, 15, 30),
        "notes": "Body excerpt",
    }

    writer.append(lead)

    assert path.exists(), "Workbook should be created"

    writer.append(lead)

    # Workbook should contain header + 2 rows
    from openpyxl import load_workbook

    wb = load_workbook(path)
    ws = wb.active
    assert ws.max_row == 3
    assert ws["A2"].value.startswith("2024-09-24")
    assert ws["B2"].value == "lead@example.com"
