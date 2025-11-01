# Email Ingestor

Email Ingestor è una piccola applicazione FastAPI che legge le email da una casella IMAP, estrae i contatti principali e li salva in un database SQLite e in un file Excel scaricabile.

## Prerequisiti

1. **Python 3.11 o superiore.**
2. Creare e attivare un ambiente virtuale:
   ```bash
   python -m venv .venv

   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1

   # macOS/Linux
   source .venv/bin/activate
   ```
3. Installare le dipendenze principali:
   ```bash
   pip install -r requirements.txt
   ```

## Configurazione (`.env`)

Copia `/.env.example` in `/.env` e imposta almeno le variabili seguenti.

### Credenziali IMAP
```env
IMAP_HOST=imap.gmail.com
IMAP_PORT=993
IMAP_USERNAME=tuoindirizzo@gmail.com
IMAP_PASSWORD=app_password_gmail
IMAP_FOLDER=INBOX
IMAP_SEARCH_SINCE_DAYS=7
```

### Percorsi di salvataggio
```env
DATABASE_URL=sqlite:///./data/app.db
EXCEL_PATH=./data/leads.xlsx
EXCEL_HEADERS=name,email,phone,company,date
```

Puoi lasciare le altre variabili ai valori di default, oppure personalizzarle secondo le note presenti in `.env.example`.

## Avviare FastAPI

Avvia l'applicazione da terminale (con l'ambiente virtuale attivo):
```bash
uvicorn app.main:app --reload
```
Il server espone l'interfaccia web all'indirizzo <http://localhost:8000/>.

## Ingestione dal pannello web

1. Apri <http://localhost:8000/> nel browser.
2. Premi **"Esegui ingestione"** per avviare la lettura della casella IMAP e il salvataggio dei lead.
3. Segui i log mostrati nella pagina per verificare l'esito dell'elaborazione.

L'esecuzione esegue automaticamente il parsing dei messaggi, elimina i duplicati, popola il database e aggiorna l'export Excel.

## Scaricare il file Excel

Dalla stessa pagina web fai clic su **"Scarica Excel"** per ottenere il file aggiornato, oppure utilizza direttamente l'endpoint dedicato:
```text
GET http://localhost:8000/export/xlsx
```

Il file viene generato nel percorso indicato da `EXCEL_PATH`.

---

Per ulteriori dettagli sui campi disponibili o su configurazioni aggiuntive consulta i commenti nel file `.env.example`.

> **Nota:** il progetto non include più file o script per l'esecuzione in Docker; utilizza l'ambiente Python locale come descritto sopra.
