from __future__ import annotations
import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker, declarative_base

# For MVP we use SQLite stored in `data/app.db`.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")

url = make_url(DATABASE_URL)
backend = url.get_backend_name()

if backend == "sqlite":
    database = url.database
    if database and ":memory:" not in database and not database.startswith("file:"):
        db_path = Path(database)
        if not db_path.is_absolute():
            db_path = Path.cwd() / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

# sqlite needs check_same_thread=False for multi-threaded FastAPI dev server
connect_args = {"check_same_thread": False} if backend == "sqlite" else {}

## crea l'engine (connessione verso il DB) e la sessione (conversazione con il DB): tiene traccia degli oggetti che creiamo/modifichiamo.
# DATABASE_URL: dove si trova il DB
# create_engine(...): crea l’engine
engine = create_engine(DATABASE_URL, future=True, echo=False, connect_args=connect_args)
# SessionLocal = sessionmaker(...): è una fabbrica che crea sessioni quando ti servono.
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
# Base = declarative_base(): classe base da cui ereditano i modelli ORM
Base = declarative_base()

## Importa i modelli (così SQLAlchemy li “conosce”) e crea le tabelle se non esistono.
def init_db():
    from .models import Contact, ContactEvent, ProcessedMessage  # noqa
    Base.metadata.create_all(bind=engine)
