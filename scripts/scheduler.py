from __future__ import annotations

import os
import time
from subprocess import run

from apscheduler.schedulers.background import BackgroundScheduler

def ingest_once():
    print("[scheduler] Running ingestion...")
    run(["python", "scripts/run_ingestor.py"], check=False)

DEFAULT_INTERVAL_MINUTES = 5


def get_scheduler_interval_minutes() -> int:
    """Return the validated interval to use for the scheduler."""

    raw_value = os.environ.get("SCHEDULER_INTERVAL_MINUTES", "").strip()
    if not raw_value:
        return DEFAULT_INTERVAL_MINUTES

    try:
        value = int(raw_value)
    except ValueError:
        print(
            f"[scheduler] Invalid SCHEDULER_INTERVAL_MINUTES='{raw_value}'. "
            f"Falling back to default ({DEFAULT_INTERVAL_MINUTES} minutes)."
        )
        return DEFAULT_INTERVAL_MINUTES

    if value < 1:
        print(
            f"[scheduler] SCHEDULER_INTERVAL_MINUTES must be >= 1 minute. "
            f"Falling back to default ({DEFAULT_INTERVAL_MINUTES} minutes)."
        )
        return DEFAULT_INTERVAL_MINUTES

    return value


## Avvia un BackgroundScheduler che ogni 5’ esegue python scripts/run_ingestor.py
## Serve per simulare un job sempre attivo senza usare cron/systemd
def main():
    interval_minutes = get_scheduler_interval_minutes()
    sched = BackgroundScheduler()
    # Run every `interval_minutes`
    sched.add_job(ingest_once, "interval", minutes=interval_minutes, id="ingest_job")
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
