from __future__ import annotations
from __future__ import annotations

from datetime import datetime
from io import BytesIO

from openpyxl import load_workbook

from libs.models import Contact


def test_contacts_endpoint_returns_data(client, session_factory):
    session = session_factory()
    try:
        contact = Contact(
            email="jane.doe@example.com",
            first_name="Jane",
            last_name="Doe",
            phone="1234567890",
            org="Example Inc",
        )
        session.add(contact)
        session.commit()

        response = client.get("/contacts")
        assert response.status_code == 200
        payload = response.json()
        assert isinstance(payload, list)
        assert len(payload) == 1
        item = payload[0]
        assert item["email"] == "jane.doe@example.com"
        assert item["first_name"] == "Jane"
        assert item["last_name"] == "Doe"
        assert item["org"] == "Example Inc"
        # created_at should be a valid ISO formatted datetime
        datetime.fromisoformat(item["created_at"])
    finally:
        session.close()


def test_export_xlsx_endpoint_returns_workbook(client, session_factory):
    session = session_factory()
    try:
        contact = Contact(
            email="john.smith@example.com",
            first_name="John",
            last_name="Smith",
            phone="555123456",
            org="Smith LLC",
        )
        session.add(contact)
        session.commit()

        response = client.get("/export/xlsx")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        assert response.headers["content-disposition"].startswith("attachment; filename=\"contacts_export.xlsx\"")

        workbook = load_workbook(BytesIO(response.content))
        sheet = workbook.active
        headers = [cell.value for cell in sheet[1]]
        assert headers == [
            "id",
            "email",
            "first_name",
            "last_name",
            "phone",
            "org",
            "source",
            "created_at",
        ]
        data_row = [cell.value for cell in sheet[2]]
        assert data_row[1] == "john.smith@example.com"
        assert data_row[2] == "John"
        assert data_row[3] == "Smith"
        assert data_row[4] == "555123456"
        assert data_row[5] == "Smith LLC"
    finally:
        session.close()
