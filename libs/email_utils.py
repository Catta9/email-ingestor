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
from bs4 import BeautifulSoup

def get_text_body(msg):
    """
    Estrae il testo leggibile dal messaggio email.
    Gestisce multipart/alternative, HTML e payload mancanti.
    """
    try:
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disp = str(part.get("Content-Disposition") or "").lower()
                # preferisci text/plain, ma accetta text/html se non trovi altro
                if ctype == "text/plain" and "attachment" not in disp:
                    payload = part.get_payload(decode=True)
                    if payload:
                        return payload.decode(part.get_content_charset() or "utf-8", errors="ignore").strip()
                elif ctype == "text/html" and "attachment" not in disp:
                    payload = part.get_payload(decode=True)
                    if payload:
                        # fallback: estrai testo dal HTML
                        soup = BeautifulSoup(payload, "html.parser")
                        return soup.get_text(separator=" ", strip=True)
        else:
            # messaggio non multipart
            payload = msg.get_payload(decode=True)
            if payload:
                return payload.decode(msg.get_content_charset() or "utf-8", errors="ignore").strip()
    except Exception as e:
        print(f"[warn] Failed to parse body: {e}")

    return ""


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
