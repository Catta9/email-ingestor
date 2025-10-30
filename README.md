# Email → CRM/Excel Ingestor (MVP)

Un tool **didattico e production-minded** che:
1) legge email da una casella IMAP (es. Gmail),
2) estrae campi chiave (nome, email, telefono, azienda) con **regex semplici**,
3) salva tutto in un **database SQLite** (estendibile a Postgres),
4) espone una piccola **API FastAPI** per consultare i contatti e
5) consente **export in Excel** con un endpoint.

> Obiettivo: imparare a progettare un sistema di ingestion end-to-end (idempotenza, parsing, storage, export).

---

## ⚙️ Setup rapido

### Requisiti
- Python 3.11+
- (Opzionale) virtualenv
- Una casella email IMAP (es. Gmail) **con accesso IMAP attivo**  
  - Gmail: preferibile **App Password** (account con 2FA) oppure IMAP via OAuth2 (non incluso in questo MVP).

### 1) Clona e crea l'env
```bash
cd /percorso/dove/vuoi
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configura l'ambiente
Copia `.env.example` in `.env` e compila con le tue credenziali IMAP.

### 3) Avvia l'API
```bash
uvicorn app.main:app --reload
```
- Healthcheck: <http://127.0.0.1:8000/health>
- Lista contatti: <http://127.0.0.1:8000/contacts>
- Export Excel: <http://127.0.0.1:8000/export/xlsx>

### 4) Esegui l'ingestor (una tantum)
```bash
python scripts/run_ingestor.py
```

### 5) (Opzione) Scheduler ogni 5 minuti
```bash
python scripts/scheduler.py
```

---

## 🧠 Concetti chiave che vedrai qui
- **Idempotenza**: non rielaboriamo due volte la stessa email (chiave = `Message-ID` o `UID` IMAP).
- **Parsing robusto**: prima regex & header extraction, in futuro NER/LLM.
- **Normalizzazione**: tabella `contacts` + `contact_events` per audit.
- **Estendibilità**: facile portare da SQLite a Postgres; facile aggiungere parsers specifici dominio.

---

## 🗄️ Struttura
```
email_ingestor_mvp/
├─ app/
│  └─ main.py           # API (FastAPI)
├─ libs/
│  ├─ db.py             # Engine, session, Base
│  ├─ models.py         # SQLAlchemy ORM (contacts, contact_events, processed_messages)
│  ├─ parser.py         # Estrazione campi (regex/header)
│  ├─ email_utils.py    # Utilità per IMAP & parsing raw email
├─ scripts/
│  ├─ run_ingestor.py   # Esegue il fetch+process
│  └─ scheduler.py      # Esegue run_ingestor periodicamente (APScheduler)
├─ data/                # SQLite DB file
├─ sample_emails/       # Esempi .eml per test locale
├─ .env.example
├─ requirements.txt
└─ README.md
```

---

## 🔄 Da SQLite a Postgres
- Cambia la `DATABASE_URL` in `libs/db.py`.
- Aggiungi Alembic per migrazioni schema.

---

## 🔐 Note sicurezza
- Non committare `.env` con credenziali reali.
- PII: tratta i dati come sensibili (export cifrati, retention, opt-out su richiesta).

---

## ✅ Roadmap
- [ ] Unit test (pytest) per parser e idempotenza
- [ ] Alembic + Postgres
- [ ] Gmail OAuth2
- [ ] UI web per revisione contatti
- [ ] Modello ML/LLM per parsing avanzato
