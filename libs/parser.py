"""Funzioni di parsing per estrarre dati anagrafici dalle email."""

from __future__ import annotations

import re
from email.utils import getaddresses
from typing import Dict, Iterable, Optional, Tuple

# ---------------------------------------------------------------------------
# Pattern principali
# ---------------------------------------------------------------------------

# Email con supporto per domini multipli e caratteri comuni (punto, trattino).
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")

# Numeri di telefono con prefissi internazionali opzionali e separatori vari.
PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{1,4}\)?[\s.-]?)?\d[\d\s().-]{6,}\d"
)

# Hint linguistici per riconoscere aziende e forme societarie.
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
    "s.a.s",
    "sas",
    "s.n.c",
    "snc",
    "inc",
    "ltd",
    "llc",
    "corp",
    "corporation",
    "gmbh",
    "sarl",
    "limited",
]

# Termini da ignorare quando compaiono come presunte aziende.
COMPANY_NOISE_TERMS = {
    "marketing",
    "vendite",
    "ufficio",
    "ufficio marketing",
    "team marketing",
    "team vendite",
    "il mio team",
    "il reparto",
    "reparto marketing",
    "dipartimento",
}

# Pattern per frasi tipiche usate dagli utenti per presentarsi.
COMPANY_PATTERNS = [
    re.compile(
        r"\bsono\s+(?P<name>[\w'’\s\.]{2,60}?)\s+(?:di|del|della|dell'|dello)\s+(?P<company>[\w&'’\.,\-\s]{3,})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bsono\s+(?P<name>[\w'’\s\.]{2,60}?)\s+e\s+lavoro\s+(?:per|presso|in)\s+(?P<company>[\w&'’\.,\-\s]{3,})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmi\s+chiamo\s+(?P<name>[\w'’\s\.]{2,60}?)\s+e\s+(?:sono|lavoro)\s+(?:per|presso|in|di|del|della|dell'|dello)\s+(?P<company>[\w&'’\.,\-\s]{3,})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\blavoro\s+(?:per|presso|in)\s+(?P<company>[\w&'’\.,\-\s]{3,})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:azienda|società|societa|company)\s+(?:si\s+chiama|chiamata|denominata)\s+(?P<company>[\w&'’\.,\-\s]{3,})",
        re.IGNORECASE,
    ),
]

# Etichette che aiutano a capire il tipo di numero di telefono trovato.
PHONE_LABEL_SCORES = {
    "cell": 5,
    "cellulare": 5,
    "mobile": 5,
    "mobil": 5,
    "tel": 3,
    "telefono": 3,
    "phone": 3,
    "ph": 3,
    "ufficio": 2,
    "office": 2,
    "lavoro": 2,
    "work": 2,
    "fax": 0,
}


# ---------------------------------------------------------------------------
# Funzioni di supporto
# ---------------------------------------------------------------------------

def normalize_whitespace(value: str) -> str:
    """Normalizza spazi multipli e ritorni a capo in una singola stringa."""

    return re.sub(r"\s+", " ", value).strip()


def set_result_name(
    result: Dict[str, Optional[str]],
    full_name: str,
    overwrite: bool = False,
) -> None:
    """Popola ``first_name`` e ``last_name`` partendo da un nome completo."""

    if not full_name:
        return

    full_name = normalize_whitespace(full_name)
    if not full_name:
        return

    titles = ["dr", "dott", "ing", "prof", "avv", "mr", "mrs", "ms", "miss", "sir"]
    parts = [p for p in full_name.lower().split() if p.rstrip(".") not in titles]

    if not parts:
        return

    first = parts[0].title()
    last = " ".join(p.title() for p in parts[1:]) if len(parts) >= 2 else None

    if overwrite or not result.get("first_name"):
        result["first_name"] = first
    if last and (overwrite or not result.get("last_name")):
        result["last_name"] = last


def _clean_company(candidate: str) -> Optional[str]:
    """Ripulisce il nome dell'azienda scartando rumore e falsi positivi."""

    candidate = normalize_whitespace(candidate).strip("-•|.,;: ")
    if len(candidate) < 3:
        return None

    lowered = candidate.lower()
    if lowered in COMPANY_NOISE_TERMS:
        return None

    if all(word.islower() for word in lowered.split()):
        if not any(char.isalpha() and char.isupper() for char in candidate):
            return None

    return candidate


def _try_extract_company_from_text(text: str, result: Dict[str, Optional[str]]) -> None:
    """Cerca pattern espliciti nel testo per compilare ``org``."""

    for pattern in COMPANY_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue

        company_raw = match.groupdict().get("company")
        company_clean = _clean_company(company_raw or "")
        if company_clean and not result.get("org"):
            result["org"] = company_clean

        name_raw = match.groupdict().get("name")
        if name_raw and not result.get("first_name"):
            set_result_name(result, name_raw)

        if company_clean:
            break


def get_header_addresses(
    headers: Dict[str, str], header_names: Iterable[str]
) -> Iterable[Tuple[str, str]]:
    """Restituisce coppie ``(nome, email)`` dagli header specificati."""

    target_names = {name.lower() for name in header_names}
    for key, value in headers.items():
        if key.lower() not in target_names or not value:
            continue
        for name, email in getaddresses([value]):
            if email:
                yield name, email


def normalize_phone(phone: str) -> Optional[str]:
    """Pulisce e standardizza un numero di telefono."""

    digits = re.sub(r"[^\d+]", "", phone)
    digit_count = len(re.sub(r"\D", "", digits))
    if digit_count < 7:
        return None

    if digits.startswith("00"):
        digits = "+" + digits[2:]

    if digits.startswith("39") and not digits.startswith("+"):
        digits = "+39" + digits[2:]
    elif re.match(r"^3\d{8,9}$", digits):
        digits = "+39" + digits

    return digits


def extract_org_from_line(line: str) -> Optional[str]:
    """Estrae il nome dell'organizzazione da una singola riga."""

    stripped = line.strip()
    if not stripped:
        return None

    lowered = stripped.lower()

    if ":" in stripped:
        before, after = stripped.split(":", 1)
        if any(h in before.lower() for h in ORG_HINTS):
            candidate = after.strip()
            if candidate:
                return candidate

    if "-" in stripped:
        before, after = (part.strip() for part in stripped.split("-", 1))
        if any(h in before.lower() for h in ORG_HINTS) and after:
            return after

    societary_pattern = r"\b(?:s\.r\.l|srl|s\.p\.a|spa|inc|ltd|llc|corp|gmbh|sarl)\b"
    if re.search(societary_pattern, lowered):
        candidate = re.sub(
            r"(?i)^(?:azienda|company|impresa|organizzazione|org|società|societa)[:\-\s]*",
            "",
            stripped,
        ).strip(" -")
        if candidate:
            return candidate

    if any(h in lowered for h in ORG_HINTS):
        candidate = re.sub(
            r"(?i)^(?:azienda|company|impresa|organizzazione|org|società|societa)[:\-\s]*",
            "",
            stripped,
        ).strip(" -")
        if candidate and len(candidate) > 3:
            return candidate

    return None


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def parse_contact_fields(
    text: str, headers: dict[str, str] | None = None
) -> Dict[str, Optional[str]]:
    """Estrae i campi principali (nome, email, telefono, azienda) da un'email."""

    headers = headers or {}
    result: Dict[str, Optional[str]] = {
        "first_name": None,
        "last_name": None,
        "email": None,
        "phone": None,
        "org": None,
    }

    # --- Email e nome dagli header -------------------------------------------------------
    for name, email in get_header_addresses(headers, ["reply-to"]):
        if email:
            result["email"] = email.strip()
        if name:
            set_result_name(result, name, overwrite=True)
        break

    if not result["email"]:
        for name, email in get_header_addresses(headers, ["from"]):
            if email:
                result["email"] = email.strip()
            if name:
                set_result_name(result, name)
            break

    if not result["email"]:
        match = EMAIL_RE.search(text)
        if match:
            result["email"] = match.group(0)

    # --- Telefono con punteggio contestuale ---------------------------------------------
    phone_candidates: list[Tuple[int, int, str]] = []  # (score, idx, numero)
    seen_numbers: set[str] = set()

    _try_extract_company_from_text(text, result)

    for idx, line in enumerate(text.splitlines()):
        matches = list(PHONE_RE.finditer(line))
        if not matches:
            continue

        lowered_line = line.lower()
        if "fax" in lowered_line and not any(h in lowered_line for h in ("cell", "mobile", "tel")):
            continue

        score = 0
        for hint, hint_score in PHONE_LABEL_SCORES.items():
            if hint in lowered_line:
                score = max(score, hint_score)

        for match in matches:
            normalized = normalize_phone(match.group(0))
            if not normalized or normalized in seen_numbers:
                continue
            seen_numbers.add(normalized)
            phone_candidates.append((score, idx, normalized))

    if not phone_candidates:
        for match in PHONE_RE.finditer(text):
            normalized = normalize_phone(match.group(0))
            if normalized and normalized not in seen_numbers:
                seen_numbers.add(normalized)
                phone_candidates.append((0, len(phone_candidates), normalized))

    if phone_candidates:
        phone_candidates.sort(key=lambda item: (-item[0], item[1]))
        result["phone"] = phone_candidates[0][2]

    # --- Organizzazione fallback ---------------------------------------------------------
    if not result.get("org"):
        for line in text.splitlines():
            org = extract_org_from_line(line)
            if org:
                result["org"] = org
                break

    return result
