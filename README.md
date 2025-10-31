# Email Ingestor – Lead Extraction & Scoring (IMAP → DB/Excel)

Automazione che legge email via **IMAP**, estrae contatti (nome, email, telefono, azienda), li salva su **SQLite** ed **Excel**, invia una **notifica SMTP** e valuta la rilevanza del messaggio con:
- **Rule-based scoring** (parole chiave / frasi negative, soglie)
- **ML Naive Bayes** (training locale su dataset JSONL)
- **Strategia ibrida** (usa ML, fallback su rule-based)

Pensato per PMI, freelance e team che vogliono **centralizzare le richieste clienti** da email e form.

---

## 🔎 Indice
- [Caratteristiche & Stack](#caratteristiche--stack)
- [Requisiti](#requisiti)
- [Configurazione (`.env`)](#configurazione-env)
  - [IMAP (lettura email)](#imap-lettura-email)
  - [Filtri mittente / keyword (pre-filtro)](#filtri-mittente--keyword-pre-filtro)
  - [SMTP (notifica)](#smtp-notifica)
  - [Database & Export](#database--export)
  - [Lead Scoring (Rule-based, ML, Hybrid)](#lead-scoring-rule-based-ml-hybrid)
  - [Logging & Scheduler](#logging--scheduler)
- [Esecuzione (end-to-end)](#esecuzione-end-to-end)
  - [Ingestor IMAP](#1-ingestor-imap)
  - [Scheduler (facoltativo)](#2-scheduler-facoltativo)
- [Export & API](#export--api)
- [Addestramento modello ML](#addestramento-modello-ml)
  - [Dataset](#dataset)
  - [Training](#training)
  - [Uso del modello](#uso-del-modello)
- [Test & CI](#test--ci)
- [Troubleshooting rapido](#troubleshooting-rapido)
- [Roadmap](#roadmap)
- [Sicurezza](#sicurezza)

---

## Caratteristiche & Stack
- **Python 3.11+**, **FastAPI** (API), **SQLAlchemy 2.x** (SQLite di default), **openpyxl** (export Excel)  
- **IMAPClient** (fetch email), parsing MIME robusto (HTML → testo con **beautifulsoup4**)  
- **SMTP** per email di riepilogo  
- **Idempotenza** (no duplicati) con tabella `processed_messages` (Message-ID/UID)  
- **Lead Scoring**: Rule-based, ML (Naive Bayes) o **Hybrid**  
- **Script di training** modello ML + salvataggio **JSON** & metriche  
- **PyTest** + **GitHub Actions** (Tests)  
- Scheduler semplice (APS) o integrazione con `cron` / Task Scheduler  

---

## Requisiti
```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

# Dipendenze
pip install -r requirements.txt
```

> `requirements.txt` include **beautifulsoup4** per il parsing HTML.

---

## Configurazione (`.env`)
Duplica `/.env.example` → `/.env` e compila le variabili.

### IMAP (lettura email)
```env
IMAP_HOST=imap.gmail.com
IMAP_PORT=993
IMAP_USERNAME=tuoindirizzo@gmail.com
IMAP_PASSWORD=app_password_gmail
IMAP_FOLDER=INBOX
IMAP_SEARCH_SINCE_DAYS=3
```
> Gmail richiede **App Password** con 2FA attiva.
- Il valore deve essere un **intero ≥ 1**. In caso di valore mancante o non valido,
  l'ingestor torna automaticamente al default di 7 giorni e scrive un log di
  avviso.

### Filtri mittente / keyword (pre-filtro)
```env
# Vuoto = accetta tutti i domini
ALLOWED_SENDER_DOMAINS=azienda.it,partner.com

KEYWORDS_INCLUDE=preventivo,richiesta,contatto
KEYWORDS_EXCLUDE=newsletter,spam,offerta
```

### SMTP (notifica)
```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=tuoindirizzo@gmail.com
SMTP_PASSWORD=app_password_gmail
SMTP_FROM="Email Ingestor <tuoindirizzo@gmail.com>"
# (Opzionale) SMTP_SENDER=...  # fallback legacy se SMTP_FROM non è impostata
NOTIFY_RECIPIENTS=me@azienda.it, collega@azienda.it
```

### Database & Export
```env
DATABASE_URL=sqlite:///./data/app.db
EXCEL_PATH=./data/leads.xlsx
EXCEL_HEADERS=name,email,phone,company,date
LEADS_XLSX_PATH=./data/leads.xlsx  # fallback per compatibilità
```

- `EXCEL_PATH` ha priorità rispetto a `LEADS_XLSX_PATH`; quest'ultimo resta per retro-compatibilità.
- `EXCEL_HEADERS` accetta una lista separata da virgole che definisce l'ordine delle colonne generate.

### Lead Scoring (Rule-based, ML, Hybrid)
```env
# rule_based | ml | hybrid
LEAD_CLASSIFIER_STRATEGY=hybrid

# Rule-based
LEAD_KEYWORDS=preventivo,richiesta,quotazione,offerta,informazioni
LEAD_NEGATIVE_KEYWORDS=non interessato,solo info,no acquisto
LEAD_SCORE_THRESHOLD=2.0

# ML
LEAD_MODEL_PATH=artifacts/lead_classifier.json
LEAD_MODEL_THRESHOLD=0.5
```

### Logging & Scheduler
```env
LOG_LEVEL=INFO
SCHEDULER_INTERVAL_MINUTES=5
```

---

## Esecuzione (end-to-end)

### 1) Ingestor IMAP
```bash
python -m scripts.run_ingestor
```

**Cosa fa:**
- Connette IMAP, legge messaggi recenti  
- Applica pre-filtri (domini/keyword)  
- Calcola **lead score** (rule-based/ML/hybrid)  
- Salva su DB + Excel  
- Invia **notifica SMTP** (se configurata)  
- Evita duplicati (Message-ID/UID)

### 2) Scheduler (facoltativo)
Esegue l’ingestor ogni `SCHEDULER_INTERVAL_MINUTES` (default `5` minuti, valori ammessi ≥ `1`).
Per impostare un intervallo personalizzato:
```bash
export SCHEDULER_INTERVAL_MINUTES=15  # esegue ogni 15 minuti
python -m scripts.scheduler
```
```bash
python -m scripts.scheduler
```

---

## Export & API
- **Excel**: file creato/aggiornato in `EXCEL_PATH` (default `./data/leads.xlsx`)
- **API FastAPI**:
  - `GET /health` → stato servizio
  - `GET /contacts?limit&offset` → lista contatti (DB)
  - `GET /export/xlsx` → scarica export Excel corrente
  - `POST /ingestion/run` → avvia ingestione IMAP manuale
  - `PATCH /contacts/{id}` → aggiorna stato (`new`/`reviewed`) e note operative
  - `POST /contacts/{id}/tags` → aggiunge un tag libero al lead (idempotente sul valore)

### Autenticazione API
Le rotte elencate sopra (eccetto `GET /health`) richiedono un header `X-API-Key`.

- Valore di default in sviluppo: `local-dev-key`
- Sovrascrivibile impostando `INGESTOR_API_KEY` (o `API_KEY`) nell'ambiente / `.env`
- La dashboard web salva la chiave nel browser (LocalStorage) e la riutilizza per tutte le chiamate

Se la chiave è errata, la UI mostra l'errore e blocca le azioni fino all'aggiornamento.

### Dashboard web
L'interfaccia `FastAPI` espone `/` con una SPA vanilla JS potenziata:

- tabella lead con stato modificabile (`Nuovo` / `In revisione`), note testuali e tag
- form rapido per aggiungere tag (evita duplicati in modo case-insensitive)
- textarea con salvataggio note e log in tempo reale delle operazioni
- card "API key" per memorizzare la chiave e collegarsi agli SSE (`/ingestion/stream`)

Esempio di schermata con lead aggiornati:

![Dashboard lead con stato, note e tag](artifacts/dashboard-leads.png)
- **Excel**: file creato/aggiornato in `EXCEL_PATH` (default `./data/leads.xlsx`)  
- **API FastAPI** (se abilitate nel progetto):
  - `GET /health` → stato servizio
  - `GET /contacts?limit&offset` → lista contatti (DB)
  - `GET /export/xlsx` → scarica export Excel corrente
  - `GET /metrics` → esporta metriche Prometheus

**Avvio API:**
```bash
uvicorn app.main:app --reload
```

## Migrazioni database
Il nuovo schema introduce i campi `status`, `notes` e la tabella relazionale `contact_tags`.

Esegui le migrazioni Alembic incluse nel repository:

```bash
alembic upgrade head
```

Per un reset o downgrade:

```bash
alembic downgrade -1
### Osservabilità & Prometheus

L'applicazione espone metriche in formato **Prometheus** all'endpoint `GET /metrics`.
I contatori principali sono:

- `email_ingestor_processed_total{folder="INBOX",domain="example.com"}` – email elaborate
- `email_ingestor_leads_total{folder="INBOX",domain="example.com"}` – lead creati
- `email_ingestor_errors_total{folder="INBOX",domain="example.com"}` – errori di ingestione
- `email_ingestor_run_discovered_messages{folder="INBOX"}` – email individuate nell'ultima scansione

Esempio di scrape config (`prometheus.yml`):

```yaml
scrape_configs:
  - job_name: email-ingestor
    metrics_path: /metrics
    static_configs:
      - targets:
          - email-ingestor.local:8000
```

Alert di base per errori consecutivi (10 minuti):

```yaml
groups:
  - name: email-ingestor
    rules:
      - alert: EmailIngestorHighErrorRate
        expr: increase(email_ingestor_errors_total[10m]) > 0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Errori di ingestione rilevati"
          description: "email_ingestor_errors_total ha registrato nuovi errori negli ultimi 10 minuti."
```

---

## Addestramento modello ML

### Dataset
Formato **JSONL** (una riga = un record):
```json
{"subject":"Richiesta preventivo","body":"Buongiorno, vorrei un preventivo...","label":1}
{"subject":"Newsletter ottobre","body":"Scopri le novità...","label":0}
```
Percorso di default: `datasets/lead_training.jsonl`

### Training
```bash
python -m scripts.train_classifier   --dataset datasets/lead_training.jsonl   --output artifacts   --test-size 0.2   --random-state 42
```

**Output:**
- `artifacts/lead_classifier.json` (modello in JSON)  
- `artifacts/lead_classifier.metrics.txt` (Accuracy/Precision/Recall/F1, TP/FP/TN/FN)

### Uso del modello
Imposta nel `.env`:
```env
LEAD_CLASSIFIER_STRATEGY=ml     # o hybrid
LEAD_MODEL_PATH=artifacts/lead_classifier.json
LEAD_MODEL_THRESHOLD=0.5
```
Poi rilancia l’ingestor.  
> **Hybrid**: prova ML; se il modello non è disponibile, usa rule-based.

---

## Docker & Docker Compose
### Immagine standalone
```bash
docker build -t email-ingestor .
docker run --env-file .env -p 8000:8000 email-ingestor
```
- Usa lo stesso `.env` del progetto per configurare IMAP, SMTP e `DATABASE_URL`.
- La cartella `model_store/` viene copiata nell'immagine: puoi sovrascriverla montando un volume se devi aggiornare i modelli.

### Docker Compose con profili
Il file `docker-compose.yml` dichiara i servizi:
- `web`: API FastAPI (porta 8000).
- `worker`: esegue `scripts.scheduler` per lanciare periodicamente l'ingestor.
- `db`: Postgres 15 con healthcheck.

Sono definiti due profili Compose per orchestrare gli ambienti:
- **Dev (`--profile dev`)**: abilita `web-dev` e `worker-dev` con `--reload` e bind mount di `app/`, `libs/` e `scripts/` per hot-reload. Esempio:
  ```bash
  docker compose --profile dev up web-dev worker-dev db
  ```
- **Prod (`--profile prod`)**: usa i servizi `web` e `worker` basati sull'immagine buildata, più `db` per Postgres. Esempio:
  ```bash
  docker compose --profile prod up -d web worker db
  ```

Entrambi i profili montano i volumi nominati `model_store` e `lead_exports` all'interno del container (`/app/model_store` e `/app/data`) per condividere i modelli ML e l'export Excel (`LEADS_XLSX_PATH`).
Compose carica automaticamente le variabili dal file `.env` (IMAP/SMTP/Postgres) tramite `env_file`. Personalizza `POSTGRES_*` e `DATABASE_URL` nel tuo `.env` per puntare al servizio `db`.

---

## Test & CI
Esecuzione locale:
```bash
pytest -v
```
La pipeline **GitHub Actions** (Tests) ora esegue:
- test unitari classici sull'host runner.
- uno smoke test Docker che builda l'immagine e lancia `pytest` dentro il container per verificare la compatibilità del runtime.

---

## Troubleshooting rapido
- **`ModuleNotFoundError: libs.db`** → lancia dalla root con `python -m scripts.run_ingestor`.  
- **Gmail: “Application-specific password required”** → usa **App Password** con 2FA attiva.  
- **`no such column ...`** dopo aggiornamenti schema → elimina `data/app.db` (ambiente locale) e rilancia per ricrearlo.  
- **Errore MIME (`multipart/alternative`)** → assicurati di avere **beautifulsoup4** installato (già in `requirements.txt`).  

---

## Roadmap
- [x] IMAP ingest, parsing MIME (HTML→testo), export Excel  
- [x] Idempotenza (ProcessedMessage), notifiche SMTP  
- [x] Rule-based scoring, **ML Naive Bayes** + training script  
- [x] Test (pytest) + CI  
- [ ] Alembic + Postgres  
- [ ] OAuth Gmail (IMAP) / Microsoft 365  
- [x] Docker Compose (app + db) e profili prod  
- [ ] UI web per revisione/annotazione lead  
- [ ] Modelli ML più avanzati (n-gram, TF-IDF, transformer leggeri)  
- [ ] Metriche & monitoraggio (Prometheus/exporter)  

---

## Sicurezza
- **Non** committare il file `.env` o credenziali reali.  
- Usa **account di test** per sviluppo ed evita di esportare PII in repository pubblici.  
- Valuta **retention** dei dati e cifratura degli export in ambienti produttivi.
