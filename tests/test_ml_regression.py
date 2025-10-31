from __future__ import annotations

import json
from pathlib import Path

import pytest

from libs.ml_classifier import LeadMLClassifier, LeadModelConfig
from scripts.evaluate_classifier import evaluate
from scripts.train_classifier import TrainingConfig, train


@pytest.fixture()
def dataset_path() -> Path:
    return Path("datasets/lead_training.jsonl").resolve()


def test_logreg_training_produces_metrics(tmp_path, monkeypatch, dataset_path):
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()

    model_path = output_dir / "lead_classifier.joblib"
    monkeypatch.setenv("LEAD_MODEL_PATH", str(tmp_path / "model_store" / "lead_classifier.joblib"))

    config = TrainingConfig(
        dataset_path=dataset_path,
        output_path=model_path,
        test_size=0.25,
        random_state=42,
        use_ngrams=True,
        use_features=True,
        augment_data=False,
        algorithm="logreg",
    )

    train(config)

    metrics_json = json.loads(model_path.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    metrics = metrics_json["metrics"]

    assert metrics["accuracy"] >= 0.7
    assert metrics["roc_auc"] >= 0.75

    classifier = LeadMLClassifier(LeadModelConfig(model_path=model_path, threshold=0.5))
    sample_headers = {"Subject": "Richiesta preventivo impianto fotovoltaico"}
    sample_body = "Buongiorno, avremmo bisogno di una quotazione dettagliata."
    score = classifier.score(sample_headers, sample_body)
    assert 0.0 <= score <= 1.0

    eval_dir = tmp_path / "evaluation"
    metrics_path = evaluate(dataset_path, model_path, eval_dir)
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "metrics" in saved_metrics
    assert saved_metrics["metrics"]["roc_auc"] >= 0.75
    assert (eval_dir / "evaluation_curves.json").exists()


def test_legacy_json_format_remains_supported(tmp_path):
    source = Path("artifacts/lead_classifier.json").resolve()
    if not source.exists():  # pragma: no cover - defensive for environments without artifact
        pytest.skip("Legacy artifact not available")

    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    classifier = LeadMLClassifier(LeadModelConfig(model_path=legacy_path, threshold=0.5))
    headers = {"Subject": "Preventivo manutenzione"}
    body = "Chiediamo un preventivo per manutenzione straordinaria."
    score = classifier.score(headers, body)
    assert 0.0 <= score <= 1.0
