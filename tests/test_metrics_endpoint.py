from fastapi.testclient import TestClient

from app.main import app
from libs.metrics import (
    CONTENT_TYPE_LATEST,
    record_email_processed,
    record_ingestion_error,
    record_lead_created,
    reset_metrics,
    set_discovered_messages,
)


def test_metrics_endpoint_returns_prometheus_format() -> None:
    reset_metrics()
    set_discovered_messages(folder="INBOX", total=5)
    record_email_processed(folder="INBOX", domain="example.com")
    record_lead_created(folder="INBOX", domain="example.com")
    record_ingestion_error(folder="INBOX", domain="example.com")

    client = TestClient(app)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers.get("content-type") == CONTENT_TYPE_LATEST

    body = response.text
    assert "email_ingestor_processed_total" in body
    assert "email_ingestor_leads_total" in body
    assert "email_ingestor_errors_total" in body
    assert "email_ingestor_run_discovered_messages" in body
    assert 'domain="example.com"' in body
    assert 'folder="INBOX"' in body
