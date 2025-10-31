"""Test per i miglioramenti implementati."""
from __future__ import annotations

import pytest
from libs.parser import parse_contact_fields, normalize_phone, extract_org_from_line
from libs.ml_classifier import _tokenize, _extract_ngrams, _extract_features, EXTENDED_STOPWORDS


# === Test Parser Migliorato ===

def test_parser_extracts_mobile_over_office():
    """Verifica che preferisca cellulare rispetto a telefono ufficio."""
    text = """
    Cordiali saluti,
    Mario Rossi
    Tel. ufficio: +39 02 1234567
    Cell: +39 345 9876543
    Email: mario.rossi@example.com
    """
    headers = {"From": "Mario Rossi <mario.rossi@example.com>"}
    result = parse_contact_fields(text, headers=headers)
    
    assert result["phone"].endswith("3459876543"), "Dovrebbe preferire cellulare"


def test_parser_skips_fax_only_lines():
    """Verifica che ignori righe con solo fax."""
    text = """
    John Doe
    Phone: +44 20 1234 5678
    Fax: +44 20 9999 9999
    """
    headers = {"From": "John Doe <john@example.com>"}
    result = parse_contact_fields(text, headers=headers)
    
    assert result["phone"].endswith("2012345678"), "Dovrebbe prendere phone, non fax"


def test_parser_normalizes_italian_mobile():
    """Verifica normalizzazione cellulari italiani."""
    assert normalize_phone("3401234567") == "+393401234567"
    assert normalize_phone("340 123 4567") == "+393401234567"
    assert normalize_phone("0039 340 1234567") == "+393401234567"


def test_parser_extracts_org_with_srl():
    """Verifica estrazione organizzazione con sigla societaria."""
    line = "Azienda: Rossi Impianti S.r.l."
    org = extract_org_from_line(line)
    assert org == "Rossi Impianti S.r.l."
    
    line = "Company - Tech Solutions LLC"
    org = extract_org_from_line(line)
    assert org == "Tech Solutions LLC"


def test_parser_removes_common_titles():
    """Verifica rimozione titoli comuni."""
    text = "Dr. Mario Rossi\nEmail: dr.mario@example.com"
    headers = {"From": "Dr. Mario Rossi <dr.mario@example.com>"}
    result = parse_contact_fields(text, headers=headers)
    
    # "Dr" dovrebbe essere rimosso
    assert result["first_name"] == "Mario"
    assert result["last_name"] == "Rossi"


# === Test ML Classifier Migliorato ===

def test_tokenize_removes_stopwords():
    """Verifica rimozione stopwords estese."""
    text = "Buongiorno, vorrei un preventivo per la mia azienda"
    tokens = _tokenize(text, EXTENDED_STOPWORDS)
    
    # Stopwords rimossi
    assert "vorrei" not in tokens
    assert "per" not in tokens
    assert "la" not in tokens
    assert "mia" not in tokens
    
    # Token utili mantenuti
    assert "buongiorno" in tokens
    assert "preventivo" in tokens
    assert "azienda" in tokens


def test_tokenize_removes_short_tokens():
    """Verifica rimozione token troppo corti."""
    text = "a bb ccc dddd"
    tokens = _tokenize(text, set())
    
    assert "a" not in tokens
    assert "bb" not in tokens
    assert "ccc" in tokens
    assert "dddd" in tokens


def test_extract_ngrams():
    """Verifica estrazione bigrams."""
    tokens = ["richiesta", "preventivo", "urgente"]
    bigrams = _extract_ngrams(tokens, n=2)
    
    assert "richiesta preventivo" in bigrams
    assert "preventivo urgente" in bigrams
    assert len(bigrams) == 2


def test_extract_features_urgency():
    """Verifica rilevamento urgenza."""
    text = "URGENTE: serve preventivo ASAP"
    features = _extract_features(text, {})
    
    assert features["urgency_score"] > 0.5


def test_extract_features_greeting():
    """Verifica rilevamento saluti."""
    text_it = "Buongiorno, vorrei informazioni"
    features_it = _extract_features(text_it, {})
    assert features_it["has_greeting"] == 1.0
    
    text_en = "Hello team, I need a quote"
    features_en = _extract_features(text_en, {})
    assert features_en["has_greeting"] == 1.0


def test_extract_features_contact_info():
    """Verifica rilevamento info contatto."""
    text = """
    Mario Rossi
    Tel: +39 340 1234567
    Email: mario@example.com
    """
    features = _extract_features(text, {})
    
    assert features["has_contact_info"] == 1.0
    assert features["signature_score"] > 0.5


def test_extract_features_question_density():
    """Verifica densità domande."""
    text_questions = "Quanto costa? Quando disponibile? Tempi di consegna?"
    features_q = _extract_features(text_questions, {})
    assert features_q["question_density"] > 0.5
    
    text_no_questions = "Vorrei un preventivo per servizi di consulenza"
    features_no = _extract_features(text_no_questions, {})
    assert features_no["question_density"] == 0.0


# === Test Data Augmentation ===

def test_augmentation_generates_variants():
    """Verifica che augmentation generi varianti."""
    from scripts.train_classifier import _augment_record
    
    record = {
        "label": 1,
        "subject": "Richiesta preventivo",
        "body": "Vorrei un preventivo urgente"
    }
    
    variants = _augment_record(record)
    
    # Almeno 2 varianti (originale + sinonimo)
    assert len(variants) >= 2
    assert variants[0] == record  # originale sempre incluso
    
    # Verifica che ci sia una variante con sinonimo
    has_synonym = any(
        "quotazione" in v.get("subject", "").lower() or 
        "quotazione" in v.get("body", "").lower()
        for v in variants[1:]
    )
    assert has_synonym


# === Test Ensemble Classifier ===

def test_ensemble_config_from_env(monkeypatch):
    """Verifica caricamento configurazione ensemble."""
    from libs.ensemble_classifier import EnsembleConfig
    
    monkeypatch.setenv("LEAD_CLASSIFIER_STRATEGY", "hybrid")
    monkeypatch.setenv("ENSEMBLE_ML_WEIGHT", "0.8")
    
    config = EnsembleConfig.from_env()
    
    assert config.strategy == "hybrid"
    assert config.ml_weight == 0.8


def test_ensemble_rule_based_fallback(monkeypatch):
    """Verifica fallback a rule-based quando ML non disponibile."""
    from libs.ensemble_classifier import EnsembleLeadClassifier
    
    monkeypatch.setenv("LEAD_CLASSIFIER_STRATEGY", "hybrid")
    monkeypatch.setenv("LEAD_MODEL_PATH", "/path/not/exists.json")
    
    # Non dovrebbe crashare, ma usare rule-based
    classifier = EnsembleLeadClassifier()
    assert not classifier.ml_available


# === Test Logging Strutturato ===

def test_structured_formatter():
    """Verifica formatter JSON."""
    import logging
    import json
    from scripts.run_ingestor import StructuredFormatter
    
    formatter = StructuredFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    record.imap_uid = "123"
    record.message_id = "<test@example.com>"
    record.esito = "ingested"
    
    formatted = formatter.format(record)
    parsed = json.loads(formatted)
    
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "Test message"
    assert parsed["imap_uid"] == "123"
    assert parsed["message_id"] == "<test@example.com>"
    assert parsed["esito"] == "ingested"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])