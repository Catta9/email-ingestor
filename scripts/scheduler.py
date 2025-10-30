from __future__ import annotations
import time
from apscheduler.schedulers.background import BackgroundScheduler
from subprocess import run

def ingest_once():
    print("[scheduler] Running ingestion...")
    run(["python", "scripts/run_ingestor.py"], check=False)

## Avvia un BackgroundScheduler che ogni 5’ esegue python scripts/run_ingestor.py
## Serve per simulare un job sempre attivo senza usare cron/systemd
def main():
    sched = BackgroundScheduler()
    # Run every 5 minutes
    sched.add_job(ingest_once, "interval", minutes=5, id="ingest_job")
    sched.start()
    print("[scheduler] Started. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[scheduler] Stopping...")
        sched.shutdown()

if __name__ == "__main__":
    main()
