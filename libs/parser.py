from __future__ import annotations

from typing import Optional, Dict, Iterable, Tuple
import re
from email.utils import parseaddr, getaddresses

## parser heuristico per estrarre campi contatto da testo email
## Regex = regular expression: un linguaggio compatto per riconoscere pattern di testo.
# EMAIL_RE cerca sequenze tipo qualcosa@dominio.estensione.
# PHONE_RE è più permissivo: supporta prefisso +39, spazi, trattini, parentesi.

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
# phone regex that captures sequences with at least 8 digits including separators
PHONE_RE = re.compile(r'(?:\+?\d[\d\s().-]{6,}\d)')
ORG_HINTS = [
    "azienda",
    "company",
    "impresa",
    "org",
    "organizzazione",
    "società",
    "societa",
    "s.r.l",
    "srl",
    "s.p.a",
    "spa",
    "inc",
    "ltd",
    "llc",
]

PHONE_LABEL_SCORES = {
    "cell": 4,
    "mobile": 4,
    "tel": 2,
    "telefono": 2,
    "phone": 2,
    "ufficio": 1,
    "office": 1,
}

def normalize_whitespace(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


def set_result_name(result: Dict[str, Optional[str]], full_name: str, overwrite: bool = False) -> None:
    if not full_name:
        return
    full_name = normalize_whitespace(full_name)
    if not full_name:
        return
    parts = full_name.split(" ")
    first = parts[0]
    last = " ".join(parts[1:]) if len(parts) >= 2 else None
    if overwrite or not result.get("first_name"):
        result["first_name"] = first
    if last and (overwrite or not result.get("last_name")):
        result["last_name"] = last


def get_header_addresses(headers: Dict[str, str], header_names: Iterable[str]) -> Iterable[Tuple[str, str]]:
    """Return parsed (name, email) tuples for the given header names (case insensitive)."""
    target_names = {name.lower() for name in header_names}
    for key, value in headers.items():
        if key.lower() in target_names and value:
            for item in getaddresses([value]):
                name, email = item
                if email:
                    yield name, email


def normalize_phone(phone: str) -> Optional[str]:
    digits = re.sub(r"[^\d+]", "", phone)
    # basic sanity check: ignore very short strings (<7 digits)
    digit_count = len(re.sub(r"\D", "", digits))
    if digit_count < 7:
        return None
    # convert leading 00 to + to normalize international prefix
    if digits.startswith("00"):
        digits = "+" + digits[2:]
    return digits

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
    from_hdr = next((v for k, v in headers.items() if k.lower() == "from"), None)
    if from_hdr:
        name, email = parseaddr(from_hdr)
        if email:
            result["email"] = email.strip()
        if name:
            set_result_name(result, name)

    # Use Reply-To name if present (takes precedence over generic From names)
    for name, _ in get_header_addresses(headers, ["reply-to"]):
        if name:
            set_result_name(result, name, overwrite=True)
            break

    # Prefer Reply-To email if present
    for _, email in get_header_addresses(headers, ["reply-to"]):
        if email:
            result["email"] = email.strip()
            break

    # If still no email, ensure we consider From (if parseaddr didn't find)
    if not result["email"]:
        for _, email in get_header_addresses(headers, ["from"]):
            if email:
                result["email"] = email.strip()
                break

    # Se email non trovata, cerca nel testo con regex (primo match)
    if not result["email"]:
        m = EMAIL_RE.search(text)
        if m:
            result["email"] = m.group(0)

    # Cerca telefono con regex valutando etichette
    phone_candidates: list[Tuple[int, int, str]] = []
    seen_numbers: set[str] = set()
    for idx, line in enumerate(text.splitlines()):
        matches = list(PHONE_RE.finditer(line))
        if not matches:
            continue
        low = line.lower()
        if "fax" in low and not any(h in low for h in ("cell", "mobile")):
            # skip fax-only lines
            continue
        score = 0
        for hint, hint_score in PHONE_LABEL_SCORES.items():
            if hint in low:
                score = max(score, hint_score)
        for m in matches:
            raw_phone = m.group(0)
            normalized = normalize_phone(raw_phone)
            if not normalized or normalized in seen_numbers:
                continue
            seen_numbers.add(normalized)
            phone_candidates.append((score, idx, normalized))

    if not phone_candidates:
        # fallback to scanning whole text if nothing found in labelled lines
        for m in PHONE_RE.finditer(text):
            normalized = normalize_phone(m.group(0))
            if normalized and normalized not in seen_numbers:
                seen_numbers.add(normalized)
                phone_candidates.append((0, len(phone_candidates), normalized))

    if phone_candidates:
        # sort by score desc, then by appearance order (idx)
        phone_candidates.sort(key=lambda item: (-item[0], item[1]))
        result["phone"] = phone_candidates[0][2]

    ## cerca a dedurre la azienda cercando righe con parole chiave (ORG_HINTS)
    # Org: scan lines for hints
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if any(h in low for h in ORG_HINTS):
            candidate = stripped
            if ":" in stripped:
                before, after = stripped.split(":", 1)
                if any(h in before.lower() for h in ORG_HINTS):
                    candidate = after.strip()
            elif "-" in stripped:
                before, after = stripped.split("-", 1)
                if any(h in before.lower() for h in ORG_HINTS):
                    candidate = after.strip()

            candidate = re.sub(
                r"(?i)^(?:azienda|company|impresa|organizzazione|org|società|societa)[:\-\s]*",
                "",
                candidate,
            ).strip(" -")
            if candidate:
                result["org"] = candidate
                break

    return result
