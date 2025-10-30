import importlib
import os
import sys


def test_sqlite_parent_directory_created(tmp_path, monkeypatch):
    db_dir = tmp_path / "nested" / "path"
    db_path = db_dir / "test.db"

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    assert not db_dir.exists()

    # ensure module is reloaded with the new environment
    sys.modules.pop("libs.db", None)
    db_module = importlib.import_module("libs.db")

    # init_db should create the file without raising
    db_module.init_db()

    assert db_dir.exists()
    assert os.path.exists(db_path)
