from __future__ import annotations
import os
import ssl
import datetime as dt
from imapclient import IMAPClient
from email import message_from_bytes
from email.policy import default as default_policy
from email.header import decode_header, make_header

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

## apre una connessione IMAP con SSL, fa login con username/password.
def get_imap_client():
    host = os.getenv("IMAP_HOST", "imap.gmail.com")
    port = env_int("IMAP_PORT", 993)
    username = os.getenv("IMAP_USERNAME")
    password = os.getenv("IMAP_PASSWORD")
    if not (username and password):
        raise RuntimeError("IMAP_USERNAME and IMAP_PASSWORD must be set in environment")

    ssl_ctx = ssl.create_default_context()
    client = IMAPClient(host, port=port, ssl=True, ssl_context=ssl_ctx)
    client.login(username, password)
    return client

def decode_mime_header(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value
## estrae e decodifica gli header utili 
def extract_headers(msg) -> dict[str, str]:
    wanted = ["Message-ID", "From", "To", "Subject", "Date"]
    headers = {}
    for w in wanted:
        v = msg[w]
        headers[w] = decode_mime_header(v) if v else None
    return headers

## estrae il corpo testo (text/plain) dall'email 
def get_text_body(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    return part.get_content().strip()
                except Exception:
                    return part.get_payload(decode=True).decode(errors="ignore")
        # fallback to first part
        part = msg.get_payload(0)
        try:
            return part.get_content().strip()
        except Exception:
            return part.get_payload(decode=True).decode(errors="ignore")
    else:
        try:
            return msg.get_content().strip()
        except Exception:
            return msg.get_payload(decode=True).decode(errors="ignore")

## seleziona la cartella (es. INBOX) e cerca le email ricevute negli ultimi N giorni
def search_since_days(client: IMAPClient, folder: str, since_days: int):
    client.select_folder(folder, readonly=False)
    since_date = (dt.date.today() - dt.timedelta(days=since_days))
    # IMAP uses DD-Mon-YYYY
    criteria = ['SINCE', since_date.strftime('%d-%b-%Y')]
    uids = client.search(criteria)
    return uids

## scarica il messaggio grezzo e lo trasforma in oggetto
def fetch_message(client: IMAPClient, uid: int):
    resp = client.fetch(uid, ["RFC822"])
    raw = resp[uid][b"RFC822"]
    msg = message_from_bytes(raw, policy=default_policy)
    return msg
