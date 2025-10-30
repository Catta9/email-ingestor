from __future__ import annotations

from typing import Optional, Dict
import re
from email.utils import parseaddr

## parser heuristico per estrarre campi contatto da testo email
## Regex = regular expression: un linguaggio compatto per riconoscere pattern di testo.
# EMAIL_RE cerca sequenze tipo qualcosa@dominio.estensione.
# PHONE_RE è più permissivo: supporta prefisso +39, spazi, trattini, parentesi.

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_RE = re.compile(r'(?:(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{2,4}\)?[\s\-\.]?)?\d{3}[\s\-\.]?\d{3,4}[\s\-\.]?\d{0,4})')
ORG_HINTS = ["azienda", "company", "impresa", "org", "organizzazione", "società"]

def normalize_whitespace(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def parse_contact_fields(text: str, headers: dict[str, str] | None = None) -> Dict[str, Optional[str]]:
    """
    Very simple heuristic parser:
    - Name from From: header
    - Email from From: header, fallback to first email in body
    - Phone via basic regex
    - Org by scanning lines for org hints
    """
    headers = headers or {}
    result = {
        "first_name": None,
        "last_name": None,
        "email": None,
        "phone": None,
        "org": None,
    }

    # Email & name from From header using email.utils.parseaddr
    from_hdr = headers.get("From") or headers.get("from")
    if from_hdr:
        name, email = parseaddr(from_hdr)
        if email:
            result["email"] = email.strip()
        if name:
            name = normalize_whitespace(name)
            parts = name.split(" ")
            if len(parts) >= 2:
                result["first_name"], result["last_name"] = parts[0], " ".join(parts[1:])
            else:
                result["first_name"] = parts[0]

    # Se email non trovata, cerca nel testo con regex
    if not result["email"]:
        m = EMAIL_RE.search(text)
        if m:
            result["email"] = m.group(0)

    # Cerca telefono con regex e lo normalizza
    m = PHONE_RE.search(text)
    if m:
        phone = m.group(0)
        # Keep only digits and plus for normalization
        phone_norm = re.sub(r"[^\d+]", "", phone)
        result["phone"] = phone_norm

    ## cerca a dedurre la azienda cercando righe con parole chiave (ORG_HINTS)
    # Org: scan lines for hints
    for line in text.splitlines():
        low = line.lower()
        if any(h in low for h in ORG_HINTS):
            # extract words after colon if any
            if ":" in line:
                candidate = line.split(":", 1)[1].strip()
                if len(candidate) >= 2:
                    result["org"] = candidate
                    break

    return result
