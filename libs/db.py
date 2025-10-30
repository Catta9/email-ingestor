from __future__ import annotations
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# For MVP we use SQLite. Switch to Postgres by setting DATABASE_URL accordingly.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")

# sqlite needs check_same_thread=False for multi-threaded FastAPI dev server
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

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
