from __future__ import annotations
import glob
from email import message_from_file
from email.policy import default as default_policy

from dotenv import load_dotenv

from libs.db import SessionLocal, init_db
from libs.models import Contact, ContactEvent, ProcessedMessage
from libs.email_utils import extract_headers, get_text_body
from libs.parser import parse_contact_fields

load_dotenv()
## Utile per imparare e provare parsing/DB senza toccare l’IMAP
## Scorre sample_emails/*.eml, estrae header/body, passa al parser, salva nel DB esattamente come farebbe l’ingestione IMAP
def main():
    init_db()
    files = glob.glob("sample_emails/*.eml")
    print(f"[local] Found {len(files)} .eml files")
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            msg = message_from_file(f, policy=default_policy)
        headers = extract_headers(msg)
        msg_id = headers.get("Message-ID") or path

        with SessionLocal() as db:
            # idempotency on msg_id
            exists = db.query(ProcessedMessage).filter_by(message_id=msg_id).first()
            if exists:
                continue

            body = get_text_body(msg)
            fields = parse_contact_fields(body, headers=headers)
            email_val = fields.get("email")
            if not email_val:
                print(f"[skip] {path}: no email extracted")
                db.add(ProcessedMessage(message_id=msg_id))
                db.commit()
                continue

            contact = db.query(Contact).filter_by(email=email_val).first()
            if not contact:
                contact = Contact(
                    email=email_val,
                    first_name=fields.get("first_name"),
                    last_name=fields.get("last_name"),
                    phone=fields.get("phone"),
                    org=fields.get("org"),
                    source="email"
                )
                db.add(contact)
                db.flush()

            ev = ContactEvent(
                contact_id=contact.id,
                event_type="email_inbound",
                payload={"headers": headers, "extracted": fields}
            )
            db.add(ev)
            db.add(ProcessedMessage(message_id=msg_id))
            db.commit()
            print(f"[ok] {path} → {email_val}")

if __name__ == "__main__":
    main()
