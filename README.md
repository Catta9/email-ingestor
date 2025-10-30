Email Ingestor – Lead Extraction & Scoring (IMAP → DB/Excel)

Automazione che legge email via IMAP, estrae contatti (nome, email, telefono, azienda), li salva su SQLite ed Excel, invia una notifica SMTP e valuta la rilevanza del messaggio con:

Rule-based scoring (parole chiave / frasi negative, soglie)

ML Naive Bayes (training locale su dataset JSONL)

Strategia ibrida (usa ML, fallback su rule-based)

Pensato per PMI, freelance e team che vogliono centralizzare richieste clienti da email e form.

Caratteristiche & Stack

Python 3.11+, FastAPI (API), SQLAlchemy 2.x (SQLite di default), openpyxl (export Excel)

IMAPClient (fetch email), parsing MIME robusto (HTML→testo con beautifulsoup4)

SMTP per email di riepilogo

Idempotenza (no duplicati) con tabella processed_messages (Message-ID/UID)

Lead Scoring: Rule-based, ML (Naive Bayes) o Hybrid

Script di training modello ML + salvataggio JSON & metriche

PyTest + GitHub Actions (Tests)

Scheduler semplice (APS) o integrazione con cron/Task Scheduler

Requisiti
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt


requirements.txt include beautifulsoup4 per il parsing HTML.

Configurazione (.env)

Duplica .env.example → .env e compila:

IMAP (lettura email)
IMAP_HOST=imap.gmail.com
IMAP_PORT=993
IMAP_USERNAME=tuoindirizzo@gmail.com
IMAP_PASSWORD=app_password_gmail
IMAP_FOLDER=INBOX
IMAP_SEARCH_SINCE_DAYS=3


Gmail richiede App Password con 2FA attiva.

Filtri mittente / keyword (pre-filtro)
ALLOWED_SENDER_DOMAINS=azienda.it,partner.com   # vuoto = accetta tutti
KEYWORDS_INCLUDE=preventivo,richiesta,contatto
KEYWORDS_EXCLUDE=newsletter,spam,offerta

SMTP (notifica)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=tuoindirizzo@gmail.com
SMTP_PASSWORD=app_password_gmail
SMTP_FROM="Email Ingestor <tuoindirizzo@gmail.com>"
NOTIFY_RECIPIENTS=me@azienda.it, collega@azienda.it

Database & Export
DATABASE_URL=sqlite:///./data/app.db
EXCEL_PATH=./data/leads.xlsx
EXCEL_HEADERS=name,email,phone,company,date

Lead Scoring (Rule-based, ML, Hybrid)
LEAD_CLASSIFIER_STRATEGY=hybrid     # rule_based | ml | hybrid

# Rule-based
LEAD_KEYWORDS=preventivo,richiesta,quotazione,offerta,informazioni
LEAD_NEGATIVE_KEYWORDS=non interessato,solo info,no acquisto
LEAD_SCORE_THRESHOLD=2.0

# ML
LEAD_MODEL_PATH=artifacts/lead_classifier.json
LEAD_MODEL_THRESHOLD=0.5

Logging & Scheduler
LOG_LEVEL=INFO
SCHEDULER_INTERVAL_MINUTES=5


Sicurezza: non committare .env. Usa account di test per sviluppo.

Esecuzione (end-to-end)
1) Ingestor IMAP
python -m scripts.run_ingestor


Cosa fa:

connette IMAP, legge messaggi recenti

applica pre-filtri (domini/keyword)

calcola lead score (rule-based/ML/hybrid)

salva su DB + Excel

invia notifica SMTP (se configurata)

evita duplicati (Message-ID/UID)

2) Scheduler (facoltativo)
python -m scripts.scheduler


Esegue l’ingestor ogni SCHEDULER_INTERVAL_MINUTES.

Export & API

Excel: file creato/aggiornato in EXCEL_PATH (default ./data/leads.xlsx)

API FastAPI (se abilitate nel progetto):

GET /health → stato servizio

GET /contacts?limit&offset → lista contatti (DB)

GET /export/xlsx → scarica export Excel corrente

Avvio API:

uvicorn app.main:app --reload

Addestramento modello ML
Dataset

Formato JSONL (una riga = un record):

{"subject":"Richiesta preventivo","body":"Buongiorno, vorrei un preventivo...","label":1}
{"subject":"Newsletter ottobre","body":"Scopri le novità...","label":0}


Percorso di default: datasets/lead_training.jsonl

Training
python -m scripts.train_classifier \
  --dataset datasets/lead_training.jsonl \
  --output artifacts \
  --test-size 0.2 \
  --random-state 42


Output:

artifacts/lead_classifier.json (modello in JSON)

artifacts/lead_classifier.metrics.txt (Accuracy/Precision/Recall/F1, TP/FP/TN/FN)

Uso del modello

Imposta nel .env:

LEAD_CLASSIFIER_STRATEGY=ml     # o hybrid
LEAD_MODEL_PATH=artifacts/lead_classifier.json
LEAD_MODEL_THRESHOLD=0.5


Poi rilancia l’ingestor.

Hybrid: prova ML; se il modello non è disponibile, usa rule-based.

Test & CI

Esecuzione locale:

pytest -v


La pipeline GitHub Actions (Tests) esegue i test su ogni push/PR.

Troubleshooting rapido

ModuleNotFoundError: libs.db → lancia con python -m scripts.run_ingestor dalla root del progetto.

Gmail: “Application-specific password required” → usa App Password con 2FA attiva.

no such column ... dopo aggiornamenti schema → elimina data/app.db (ambiente locale) e rilancia per ricrearlo.

Errore MIME (multipart/alternative) → assicurati di avere beautifulsoup4 installato (già in requirements.txt).

Roadmap

 IMAP ingest, parsing MIME (HTML→testo), export Excel

 Idempotenza (ProcessedMessage), notifiche SMTP

 Rule-based scoring, ML Naive Bayes + training script

 Test (pytest) + CI

 Alembic + Postgres

 OAuth Gmail (IMAP) / Microsoft 365

 Docker Compose (app + db) e profili prod

 UI web per revisione/annotazione lead

 Modelli ML più avanzati (n-gram, TF-IDF, transformer leggeri)

 Metrica/monitoraggio (prometheus/exporter)