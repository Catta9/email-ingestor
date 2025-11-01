from __future__ import annotations

from datetime import datetime
from io import BytesIO

from openpyxl import load_workbook

from libs.models import Contact, ContactTag


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
        assert item["status"] == "new"
        assert item["notes"] is None
        assert item["tags"] == []
        # created_at should be a valid ISO formatted datetime
        datetime.fromisoformat(item["created_at"])
        assert "last_message_subject" in item
        assert "last_message_received_at" in item
        assert "last_message_excerpt" in item
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
            "status",
            "tags",
            "notes",
            "last_message_subject",
            "last_message_received_at",
            "last_message_excerpt",
        ]
        data_row = [cell.value for cell in sheet[2]]
        assert data_row[1] == "john.smith@example.com"
        assert data_row[2] == "John"
        assert data_row[3] == "Smith"
        assert data_row[4] == "555123456"
        assert data_row[5] == "Smith LLC"
        assert data_row[8] == "new"
        assert data_row[9] is None
    finally:
        session.close()


def test_update_contact_status_and_notes(client, session_factory):
    session = session_factory()
    try:
        contact = Contact(email="lead@example.com", first_name="Lead")
        session.add(contact)
        session.commit()

        response = client.patch(
            f"/contacts/{contact.id}",
            json={"status": "reviewed", "notes": "Chiamato il cliente"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "reviewed"
        assert payload["notes"] == "Chiamato il cliente"

        session.refresh(contact)
        assert contact.status == "reviewed"
        assert contact.notes == "Chiamato il cliente"
    finally:
        session.close()
def test_add_tag_to_contact(client, session_factory):
    session = session_factory()
    try:
        contact = Contact(email="tag@example.com")
        session.add(contact)
        session.commit()

        response = client.post(
            f"/contacts/{contact.id}/tags",
            json={"tag": "priorità"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert "priorità" in payload["tags"]

        session.refresh(contact)
        assert any(tag.tag == "priorità" for tag in contact.tags)
    finally:
        session.close()


def test_add_tag_avoids_duplicates(client, session_factory):
    session = session_factory()
    try:
        contact = Contact(email="dup@example.com")
        contact.tags.append(ContactTag(tag="caldo"))
        session.add(contact)
        session.commit()

        response = client.post(
            f"/contacts/{contact.id}/tags",
            json={"tag": "Caldo"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["tags"] == ["caldo"]

        session.refresh(contact)
        assert [tag.tag for tag in contact.tags] == ["caldo"]
    finally:
        session.close()
