from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from urllib.parse import urlencode, urlparse

import typing as _t

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


if "bs4" not in sys.modules:
    class _FallbackBeautifulSoup:
        def __init__(self, raw_html: bytes | str, parser: str = "html.parser") -> None:
            if isinstance(raw_html, bytes):
                text = raw_html.decode("utf-8", errors="ignore")
            else:
                text = raw_html or ""
            self._text = text

        def get_text(self, separator: str = " ", strip: bool = False) -> str:
            import re

            text = re.sub(r"<[^>]+>", " ", self._text)
            text = re.sub(r"\s+", " ", text)
            if strip:
                text = text.strip()
            if separator != " ":
                return separator.join([part for part in text.split(" ") if part])
            return text

    bs4_stub = ModuleType("bs4")
    bs4_stub.BeautifulSoup = _FallbackBeautifulSoup  # type: ignore[attr-defined]
    sys.modules["bs4"] = bs4_stub


try:  # pragma: no cover - exercised implicitly in environments lacking httpx
    import httpx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    httpx = ModuleType("httpx")
    sys.modules["httpx"] = httpx

    class _Headers:
        def __init__(self, data: _t.Iterable[_t.Tuple[str, str]] | dict[str, str] | None = None) -> None:
            items: list[tuple[str, str]] = []
            if isinstance(data, dict):
                items.extend((str(k), str(v)) for k, v in data.items())
            elif data:
                items.extend((str(k), str(v)) for k, v in data)
            self._items = items

        def multi_items(self) -> list[tuple[str, str]]:
            return list(self._items)

        def items(self) -> list[tuple[str, str]]:
            return list(self._items)

        def get(self, key: str, default: str | None = None) -> str | None:
            key_lower = key.lower()
            for k, v in reversed(self._items):
                if k.lower() == key_lower:
                    return v
            return default

        def setdefault(self, key: str, value: str) -> str:
            existing = self.get(key)
            if existing is None:
                self._items.append((key, value))
                return value
            return existing

        def __getitem__(self, key: str) -> str:
            result = self.get(key)
            if result is None:
                raise KeyError(key)
            return result

        def __contains__(self, key: object) -> bool:
            if not isinstance(key, str):
                return False
            return self.get(key) is not None

        def __iter__(self):
            return iter(self._items)

    class _URL:
        def __init__(self, url: str) -> None:
            parsed = urlparse(url)
            self.scheme = parsed.scheme or "http"
            netloc = parsed.netloc or "testserver"
            self.netloc = netloc.encode("ascii")
            path = parsed.path or "/"
            self.path = path
            self.raw_path = path.encode("ascii")
            self.query = (parsed.query or "").encode("ascii")

    class ByteStream:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def read(self) -> bytes:
            return self._data

    class BaseTransport:
        def handle_request(self, request: "Request") -> "Response":  # pragma: no cover - interface only
            raise NotImplementedError

    class Request:
        def __init__(
            self,
            method: str,
            url: str,
            headers: dict[str, str] | list[tuple[str, str]] | None = None,
            content: bytes | str | None = None,
            params: dict[str, _t.Any] | None = None,
        ) -> None:
            if params:
                query = urlencode(params, doseq=True)
                separator = "&" if "?" in url else "?"
                url = f"{url}{separator}{query}" if query else url
            self.method = method.upper()
            self.url = _URL(url)
            self.headers = _Headers(headers)
            if content is None:
                body = b""
            elif isinstance(content, (bytes, bytearray)):
                body = bytes(content)
            else:
                body = str(content).encode("utf-8")
            self._body = body

        def read(self) -> bytes:
            return self._body

    class Response:
        def __init__(
            self,
            status_code: int,
            headers: list[tuple[str, str]] | dict[str, str] | None = None,
            stream: ByteStream | None = None,
            request: Request | None = None,
        ) -> None:
            self.status_code = status_code
            self.headers = _Headers(headers)
            self.request = request
            self._stream = stream or ByteStream(b"")
            self._content = self._stream.read()

        @property
        def content(self) -> bytes:
            return self._content

        def json(self) -> _t.Any:
            return json.loads(self._content.decode("utf-8"))

    class Client:
        def __init__(
            self,
            *,
            base_url: str = "http://testserver",
            headers: dict[str, str] | None = None,
            transport: BaseTransport | None = None,
            follow_redirects: bool | None = None,
            cookies: _t.Any = None,
            **kwargs: _t.Any,
        ) -> None:
            if transport is None:
                raise ValueError("transport is required for the lightweight httpx stub")
            self.base_url = base_url.rstrip("/")
            self.headers = _Headers(headers)
            self._transport = transport
            self.follow_redirects = follow_redirects
            self.cookies = cookies

        def _merge_url(self, url: str) -> str:
            url = str(url)
            if url.startswith("http://") or url.startswith("https://"):
                return url
            if not url.startswith("/"):
                url = "/" + url
            return f"{self.base_url}{url}"

        def request(self, method: str, url: str, *, headers: dict[str, str] | None = None, params: dict[str, _t.Any] | None = None, content: bytes | str | None = None, json: _t.Any = None, **kwargs: _t.Any) -> Response:
            merged_url = self._merge_url(url)
            header_items = self.headers.multi_items()
            if headers:
                if isinstance(headers, dict):
                    header_items.extend((str(k), str(v)) for k, v in headers.items())
                else:
                    header_items.extend((str(k), str(v)) for k, v in headers)
            if json is not None and content is None:
                content = json_module.dumps(json, separators=(",", ":"))  # type: ignore[name-defined]
                # ensure content-type header exists
                header_items.append(("content-type", "application/json"))
            request = Request(method, merged_url, headers=header_items, content=content, params=params)
            response = self._transport.handle_request(request)
            return response

        def get(self, url: str, *, params: dict[str, _t.Any] | None = None, headers: dict[str, str] | None = None, **kwargs: _t.Any) -> Response:
            return self.request("GET", url, params=params, headers=headers, **kwargs)

        def close(self) -> None:  # pragma: no cover - nothing to release
            return None

        def __enter__(self) -> "Client":  # pragma: no cover - not used directly in tests
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - not used directly in tests
            self.close()

    httpx.Headers = _Headers
    httpx.Request = Request
    httpx.Response = Response
    httpx.Client = Client
    httpx.BaseTransport = BaseTransport
    httpx.ByteStream = ByteStream

    types_module = ModuleType("httpx._types")
    for name in [
        "CookieTypes",
        "HeaderTypes",
        "QueryParamTypes",
        "RequestContent",
        "RequestFiles",
        "URLTypes",
        "AuthTypes",
        "TimeoutTypes",
    ]:
        setattr(types_module, name, _t.Any)
    sys.modules["httpx._types"] = types_module
    httpx._types = types_module  # type: ignore[attr-defined]

    client_module = ModuleType("httpx._client")

    class UseClientDefault:  # pragma: no cover - metadata only
        pass

    client_module.UseClientDefault = UseClientDefault
    client_module.USE_CLIENT_DEFAULT = UseClientDefault()
    sys.modules["httpx._client"] = client_module
    httpx._client = client_module  # type: ignore[attr-defined]

    # local json module reference for stub without polluting namespace
    json_module = json


def _reload_modules():
    modules = ["libs.db", "libs.models", "libs.ingestor", "app.main"]
    for name in modules:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
        else:
            importlib.import_module(name)


@pytest.fixture
def session_factory(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("INGESTOR_API_KEY", "test-key")
    _reload_modules()
    from libs.db import init_db, SessionLocal

    init_db()
    return SessionLocal


@pytest.fixture
def session(session_factory):
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client(session_factory):
    from fastapi.testclient import TestClient
    from app.main import app

    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        test_client.close()
