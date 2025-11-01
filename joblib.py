"""Lightweight drop-in replacements for the parts of joblib used in tests.

This stub implements only ``dump`` and ``load`` which are enough for the
training/evaluation pipeline.  It stores the pickled payload atomically by
writing to a temporary file and then renaming it.
"""
from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

__all__ = ["dump", "load"]


def _open_target(path: os.PathLike[str] | str, mode: str) -> BinaryIO:
    if "b" not in mode:
        mode += "b"
    return open(path, mode)


def dump(obj: Any, filename: os.PathLike[str] | str) -> None:
    """Serialize *obj* to *filename* using :mod:`pickle`.

    The real ``joblib`` library offers more features (compression, caching,
    parallel backends …).  For the unit tests we only need a reliable way to
    persist the trained model, so a thin pickle based implementation keeps the
    code self-contained and avoids an optional runtime dependency.
    """

    target = Path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        pickle.dump(obj, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path = Path(tmp.name)

    tmp_path.replace(target)


def load(filename: os.PathLike[str] | str) -> Any:
    """Load an object that was previously stored with :func:`dump`."""

    with _open_target(filename, "rb") as handle:
        return pickle.load(handle)
