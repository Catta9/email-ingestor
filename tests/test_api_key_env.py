from __future__ import annotations

import importlib
import asyncio
from pathlib import Path

from fastapi import Request


def test_expected_api_key_reads_from_dotenv(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    env_file = project_root / ".env"

    existing_content = env_file.read_text() if env_file.exists() else None

    try:
        env_file.write_text("INGESTOR_API_KEY=dot-env-key\n")
        monkeypatch.delenv("INGESTOR_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)

        import app.main as main_module

        reloaded = importlib.reload(main_module)
        assert reloaded._expected_api_key() == "dot-env-key"
    finally:
        if existing_content is None:
            if env_file.exists():
                env_file.unlink()
        else:
            env_file.write_text(existing_content)
        import app.main as main_module
        importlib.reload(main_module)


def test_expected_api_key_strips_whitespace(monkeypatch):
    monkeypatch.setenv("INGESTOR_API_KEY", "  trimmed-key  ")
    monkeypatch.delenv("API_KEY", raising=False)

    import app.main as main_module

    reloaded = importlib.reload(main_module)
    assert reloaded._expected_api_key() == "trimmed-key"


def test_api_key_guard_tolerates_extra_spaces(monkeypatch):
    monkeypatch.setenv("INGESTOR_API_KEY", "expected-key")
    monkeypatch.delenv("API_KEY", raising=False)

    import app.main as main_module

    reloaded = importlib.reload(main_module)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/ingestion/stream",
        "headers": [],
        "query_string": b"",
    }
    request = Request(scope)

    async def run_guard():
        await reloaded.api_key_guard(request, x_api_key=" expected-key ")

    asyncio.run(run_guard())
