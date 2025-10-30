from __future__ import annotations
import os
from contextlib import suppress

from dotenv import load_dotenv

from libs.db import SessionLocal, init_db
from libs.models import ProcessedMessage
from libs.email_utils import get_imap_client, search_since_days, fetch_message, extract_headers, get_text_body
from libs.ingestor import process_incoming_email

load_dotenv()

def allowed_sender(headers: dict) -> tuple[bool, str | None]:
    raw_from = headers.get("From") or ""
    _, email_address = parseaddr(raw_from)
    email_address = email_address.strip()

    if not email_address or "@" not in email_address:
        return False, f"Invalid sender address: {raw_from!r}"

    domain = email_address.rsplit("@", 1)[1].lower()

    allowed = os.getenv("ALLOWED_SENDER_DOMAINS", "").strip()
    if not allowed:
        return True, None

    allowed_domains = {d.strip().lower() for d in allowed.split(",") if d.strip()}
    if domain in allowed_domains:
        return True, None

    return False, f"Sender domain not allowed: {domain}"

## è lo script che userà prod dev
def main():
    ## assicura che ci siano le tabelle
    init_db()
    folder = os.getenv("IMAP_FOLDER", "INBOX")
    since_days = int(os.getenv("IMAP_SEARCH_SINCE_DAYS", "7"))

    client = get_imap_client()
    ## apre IMAP e cerca i messaggi recenti
    try:
        uids = search_since_days(client, folder, since_days)
        print(f"[ingestor] Found {len(uids)} messages since {since_days} days in '{folder}'")

        ## per ogni UID: estrae header e Message-ID -> controlla idempotenza ->estrae corpo e lo manda al parser
        for uid in uids:
            msg = fetch_message(client, uid)
            headers = extract_headers(msg)
            msg_id = (headers.get("Message-ID") or "").strip()

            with SessionLocal() as db:
                # Idempotency check on Message-ID (preferred) or IMAP UID
                already = False
                if msg_id:
                    already = db.execute(select(ProcessedMessage).where(ProcessedMessage.message_id == msg_id)).scalar_one_or_none() is not None
                if not already:
                    # fallback to UID check
                    already = db.execute(select(ProcessedMessage).where(ProcessedMessage.imap_uid == str(uid))).scalar_one_or_none() is not None

                if already:
                    continue

                is_allowed, reason = allowed_sender(headers)
                if not is_allowed:
                    skip_reason = reason or f"Sender not allowed: {headers.get('From')}"
                    print(f"[skip] {skip_reason}")
                if not allowed_sender(headers):
                    print(f"[skip] Sender not allowed: {headers.get('From')}")
                    # still mark as processed to avoid re-check
                    db.add(ProcessedMessage(message_id=msg_id or f"uid:{uid}", imap_uid=str(uid)))
                    db.commit()
                    continue

                body = get_text_body(msg)
                result = process_incoming_email(db, headers=headers, body=body, imap_uid=uid)

                if result["status"] == "processed":
                    email_val = headers.get("From") or result.get("contact_id")
                    print(f"[ok] Processed UID {uid} ({email_val})")
                else:
                    print(f"[skip] UID {uid}: {result['reason']}")

    finally:
        with suppress(Exception):
            client.logout()

if __name__ == "__main__":
    main()
