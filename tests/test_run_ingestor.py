import json
import logging

from sqlalchemy import select

from scripts import run_ingestor
from scripts import train_classifier


def test_allowed_sender_accepts_configured_domain(monkeypatch):
    monkeypatch.setenv("ALLOWED_SENDER_DOMAINS", "example.com")
    is_allowed, reason = run_ingestor.allowed_sender({"From": '"User" <USER@Example.COM>'})
    assert is_allowed is True
    assert reason is None


def test_allowed_sender_rejects_unlisted_domain(monkeypatch):
    monkeypatch.setenv("ALLOWED_SENDER_DOMAINS", "example.com")
    is_allowed, reason = run_ingestor.allowed_sender({"From": "person@another.com"})
    assert is_allowed is False
    assert reason == "Sender domain not allowed: another.com"


def test_allowed_sender_handles_invalid_addresses(monkeypatch):
    monkeypatch.setenv("ALLOWED_SENDER_DOMAINS", "example.com")
    is_allowed, reason = run_ingestor.allowed_sender({"From": "not-an-email"})
    assert is_allowed is False
    assert "Invalid sender address" in reason


def test_runner_logs_domain_skip(session_factory, caplog):
    headers = {
        "From": '"Blocked" <blocked@blocked.com>',
        "Message-ID": "<blocked@example.com>",
    }

    session_cls = session_factory
    db = session_cls()
    try:
        with caplog.at_level(logging.INFO, logger="scripts.run_ingestor"):
            run_ingestor._mark_sender_disallowed(
                db,
                message_id=headers["Message-ID"],
                uid_str="500",
                from_domain="blocked.com",
                reason="Sender domain not allowed: blocked.com",
            )
    finally:
        db.close()

    records = [record for record in caplog.records if getattr(record, "esito", "") == "skipped"]
    assert records, "Expected skip log entry"
    record = records[-1]
    assert record.imap_uid == "500"
    assert record.message_id == "<blocked@example.com>"
    assert record.from_domain == "blocked.com"

    with session_cls() as verify_db:
        stored = verify_db.execute(select(run_ingestor.ProcessedMessage)).scalars().all()
    assert stored, "ProcessedMessage should be recorded"


def _reset_classifiers():
    run_ingestor._RULE_BASED_SCORER = None
    run_ingestor._ML_CLASSIFIER = None
    run_ingestor._ML_AVAILABLE = True


def _write_dataset(path, records):
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines), encoding="utf-8")


def _train_ml_model(monkeypatch, tmp_path, records):
    dataset_path = tmp_path / "dataset.jsonl"
    _write_dataset(dataset_path, records)

    output_dir = tmp_path / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LEAD_MODEL_PATH", str(output_dir / "copy.json"))

    config = train_classifier.TrainingConfig(
        dataset_path=dataset_path,
        output_path=output_dir / "lead_classifier.json",
        test_size=0.25,
        random_state=42,
    )
    train_classifier.train(config)
    return config.output_path


def test_lead_scorer_detects_linguistic_variants(monkeypatch):
    monkeypatch.delenv("LEAD_KEYWORDS", raising=False)
    monkeypatch.setenv("LEAD_SCORE_THRESHOLD", "2.0")
    monkeypatch.delenv("LEAD_NEGATIVE_KEYWORDS", raising=False)
    monkeypatch.setenv("LEAD_CLASSIFIER_STRATEGY", "rule_based")
    _reset_classifiers()

    headers = {"Subject": "Richiesta preventivi"}
    assert run_ingestor.matches_lead_keywords(headers, "") is True

    _reset_classifiers()
    headers = {"Subject": "Quote request for services"}
    assert run_ingestor.matches_lead_keywords(headers, "") is True

    _reset_classifiers()
    headers = {"Subject": "Richiesta informazioni"}
    body = "Serve una quotazione urgente"
    assert run_ingestor.matches_lead_keywords(headers, body) is False


def test_lead_scorer_handles_negative_keywords(monkeypatch):
    monkeypatch.delenv("LEAD_KEYWORDS", raising=False)
    monkeypatch.setenv("LEAD_SCORE_THRESHOLD", "2.0")
    monkeypatch.setenv("LEAD_NEGATIVE_KEYWORDS", "non serve preventivo")
    monkeypatch.setenv("LEAD_CLASSIFIER_STRATEGY", "rule_based")
    _reset_classifiers()

    headers = {"Subject": "Richiesta preventivo"}
    body = "In realta non serve preventivo al momento"
    assert run_ingestor.matches_lead_keywords(headers, body) is False


def test_lead_scorer_combines_subject_and_body(monkeypatch):
    monkeypatch.delenv("LEAD_KEYWORDS", raising=False)
    monkeypatch.setenv("LEAD_SCORE_THRESHOLD", "2.0")
    monkeypatch.delenv("LEAD_NEGATIVE_KEYWORDS", raising=False)
    monkeypatch.setenv("LEAD_CLASSIFIER_STRATEGY", "rule_based")
    _reset_classifiers()

    headers = {"Subject": "Richiesta informazioni"}
    body = "Vorremmo una quotazione e un'offerta dettagliata"
    assert run_ingestor.matches_lead_keywords(headers, body) is True

    _reset_classifiers()
    body = "Vorremmo solo una quotazione"
    assert run_ingestor.matches_lead_keywords(headers, body) is False


def test_ml_classifier_strategy(monkeypatch, tmp_path):
    records = [
        {"label": 1, "subject": "Richiesta preventivo", "body": "Preventivo per sito web aziendale"},
        {"label": 1, "subject": "Quote request", "body": "Richiesta quotazione software gestionale"},
        {"label": 0, "subject": "Newsletter aziendale", "body": "Promo speciali del mese"},
        {"label": 0, "subject": "Aggiornamento manutenzione", "body": "Il servizio sarà sospeso domenica"},
        {"label": 1, "subject": "Preventivo", "body": "Vorremmo conoscere il prezzo"},
        {"label": 0, "subject": "Report mensile", "body": "Trovi allegato il report"},
    ]
    model_path = _train_ml_model(monkeypatch, tmp_path, records)

    monkeypatch.setenv("LEAD_CLASSIFIER_STRATEGY", "ml")
    monkeypatch.setenv("LEAD_MODEL_PATH", str(model_path))
    monkeypatch.setenv("LEAD_MODEL_THRESHOLD", "0.7")
    _reset_classifiers()

    headers = {"Subject": "Richiesta preventivo"}
    assert run_ingestor.matches_lead_keywords(headers, "Preventivo per sito web") is True

    headers = {"Subject": "Newsletter aziendale"}
    assert run_ingestor.matches_lead_keywords(headers, "Promo speciali del mese") is False


def test_hybrid_strategy_falls_back(monkeypatch, tmp_path):
    records = [
        {"label": 1, "subject": "Richiesta preventivo", "body": "Preventivo impianto"},
        {"label": 1, "subject": "Offerta commerciale", "body": "Vorremmo una quotazione"},
        {"label": 0, "subject": "Newsletter settimanale", "body": "Resta aggiornato"},
        {"label": 0, "subject": "Aggiornamento manutenzione", "body": "Servizio sospeso"},
    ]
    model_path = _train_ml_model(monkeypatch, tmp_path, records)

    monkeypatch.setenv("LEAD_CLASSIFIER_STRATEGY", "hybrid")
    monkeypatch.setenv("LEAD_MODEL_PATH", str(model_path))
    monkeypatch.setenv("LEAD_MODEL_THRESHOLD", "0.95")
    monkeypatch.delenv("LEAD_NEGATIVE_KEYWORDS", raising=False)
    monkeypatch.setenv("LEAD_SCORE_THRESHOLD", "2.0")
    _reset_classifiers()

    headers = {"Subject": "Richiesta informazioni"}
    body = "Vorremmo una quotazione e un'offerta dettagliata"
    assert run_ingestor.matches_lead_keywords(headers, body) is True
