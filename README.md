# Email → CRM/Excel Ingestor (MVP)

Un tool **didattico e production-minded** che:
1) legge email da una casella IMAP (es. Gmail),
2) estrae campi chiave (nome, email, telefono, azienda) con **regex semplici**,
3) salva tutto in un **database SQLite** (estendibile a Postgres),
4) aggiorna automaticamente un **file Excel** (`data/leads.xlsx`) pronto per l'import in CRM/Google Sheets,
5) espone una piccola **API FastAPI** per consultare i contatti e
6) consente **export in Excel** con un endpoint.

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
Ogni nuovo lead (matching parole chiave) viene salvato nel DB e nel file `data/leads.xlsx`.

### 5) (Opzione) Scheduler ogni 5 minuti
```bash
python scripts/scheduler.py
```

### 6) Verifica automatica con pytest
```bash
pytest
```
La suite copre API, ingestion, parser, writer Excel e notifier: deve passare interamente in locale prima di consegnare modifiche.

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
│  ├─ ingestor.py       # Logica di idempotenza + salvataggio contatti
│  ├─ lead_storage.py   # Append automatico al file Excel dei lead
│  └─ notifier.py       # Invio mail di notifica (facoltativo)
├─ scripts/
│  ├─ run_ingestor.py   # Esegue il fetch+process (filtra keyword, aggiorna Excel, invia notifiche)
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

## 📬 Configurazione avanzata
- **Parole chiave lead**: variabile `LEAD_KEYWORDS` (lista separata da virgola). Default: `preventivo, quotazione, prezzo, offerta, proposal, estimate`. Il classificatore espande automaticamente plurali, sinonimi comuni e frasi multi-parola (es. "quote request", "stima costi").
- **Soglia scoring**: `LEAD_SCORE_THRESHOLD` (float, default `2.0`) regola quanto deve essere alta la somma pesata di subject/body/header per considerare l'email un lead.
- **Keyword negative**: `LEAD_NEGATIVE_KEYWORDS` (lista separata da virgola) permette di escludere contesti come "non serve preventivo" o "solo informazioni generiche".
- **File Excel**: `LEADS_XLSX_PATH` per scegliere il percorso di output (default `data/leads.xlsx`).
- **Notifiche email** (facoltative): imposta `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_SENDER` e `NOTIFY_RECIPIENTS` per ricevere un alert ogni nuovo lead.

## 🔍 Algoritmo di scoring lead
Il modulo `LeadRelevanceScorer` normalizza subject, corpo e intestazioni (lowercase, stopword removal, stemming leggero per pattern come `preventiv`, `quotaz`, `offert`). Ogni segmento contribuisce allo score con pesi differenti (subject > corpo > intestazioni) e vengono riconosciute sia keyword singole sia frasi multi-parola/sinonimi configurati. La presenza di keyword negative annulla lo score. L'email è considerata lead solo se lo score finale supera `LEAD_SCORE_THRESHOLD`.

---

## 🔐 Note sicurezza
- Non committare `.env` con credenziali reali.
- PII: tratta i dati come sensibili (export cifrati, retention, opt-out su richiesta).

---

## ✅ Roadmap
- [x] Unit test (pytest) per parser e idempotenza (✔ `pytest`)
- [ ] Alembic + Postgres
- [ ] Gmail OAuth2
- [ ] UI web per revisione contatti
- [ ] Modello ML/LLM per parsing avanzato

## 🔁 Piano transizione FastAPI lifespan
Per eliminare il warning su `@app.on_event("startup")` e preparare l'API a FastAPI 1.0:

1. **Mappare gli hook esistenti.** In `app/main.py` oggi chiamiamo `init_db()` dentro l'evento `startup`. Documentare qualsiasi altra inizializzazione nascosta nei moduli importati.
2. **Introdurre un lifespan context.** Convertire l'app in:
   ```python
   app = FastAPI(title="Email → CRM/Excel Ingestor", lifespan=lifespan)
   ```
   dove `lifespan` è un async context manager che richiama `init_db()` nella sezione `yield`.
3. **Aggiornare i test.** Garantire che i test FastAPI utilizzino `AsyncClient`/`LifespanManager` o l'opzione `lifespan="on"` per inizializzare il contesto.
4. **Rimuovere gli eventi deprecati.** Eliminare `@app.on_event` e il warning scomparirà mantenendo la compatibilità futura.
