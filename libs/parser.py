from __future__ import annotations

from typing import Optional, Dict, Iterable, Tuple
import re
from email.utils import parseaddr, getaddresses

# Regex migliorata
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
# Phone con supporto internazionale esteso
PHONE_RE = re.compile(r'(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{1,4}\)?[\s.-]?)?\d[\d\s().-]{6,}\d')

ORG_HINTS = [
    "azienda", "company", "impresa", "org", "organizzazione", "società", "societa",
    "s.r.l", "srl", "s.p.a", "spa", "s.a.s", "sas", "s.n.c", "snc",
    "inc", "ltd", "llc", "corp", "corporation", "gmbh", "sarl", "limited",
]

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

# Pattern contestuali per telefono (label: priorità)
PHONE_LABEL_SCORES = {
    "cell": 5, "cellulare": 5, "mobile": 5, "mobil": 5,
    "tel": 3, "telefono": 3, "phone": 3, "ph": 3,
    "ufficio": 2, "office": 2, "lavoro": 2, "work": 2,
    "fax": 0,  # ignora fax
}


def normalize_whitespace(s: str) -> str:
    """Normalizza spazi multipli."""
    return re.sub(r'\s+', ' ', s).strip()


def set_result_name(
    result: Dict[str, Optional[str]],
    full_name: str,
    overwrite: bool = False
) -> None:
    """Estrae first_name e last_name da nome completo."""
    if not full_name:
        return
    full_name = normalize_whitespace(full_name)
    if not full_name:
        return
    
    # Rimuovi titoli comuni
    titles = ['dr', 'dott', 'ing', 'prof', 'avv', 'mr', 'mrs', 'ms', 'miss', 'sir']
    parts = full_name.lower().split()
    parts = [p for p in parts if p.rstrip('.') not in titles]
    
    if not parts:
        return
    
    first = parts[0].title()
    last = " ".join(p.title() for p in parts[1:]) if len(parts) >= 2 else None
    
    if overwrite or not result.get("first_name"):
        result["first_name"] = first
    if last and (overwrite or not result.get("last_name")):
        result["last_name"] = last


def _clean_company(candidate: str) -> Optional[str]:
    candidate = normalize_whitespace(candidate)
    candidate = candidate.strip("-•|.,;: ")
    if len(candidate) < 3:
        return None
    low = candidate.lower()
    if low in COMPANY_NOISE_TERMS:
        return None
    if all(word.islower() for word in low.split()):
        # Richiedi almeno un carattere maiuscolo per evitare reparti generici
        if not any(char.isalpha() and char.isupper() for char in candidate):
            return None
    return candidate


def _try_extract_company_from_text(text: str, result: Dict[str, Optional[str]]) -> None:
    for pattern in COMPANY_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        company_raw = match.groupdict().get("company")
        if not company_raw:
            continue
        company_clean = _clean_company(company_raw)
        if not company_clean:
            continue
        if not result.get("org"):
            result["org"] = company_clean

        name_raw = match.groupdict().get("name")
        if name_raw and not result.get("first_name"):
            set_result_name(result, name_raw)
        break


def get_header_addresses(
    headers: Dict[str, str], 
    header_names: Iterable[str]
) -> Iterable[Tuple[str, str]]:
    """Ritorna tuple (nome, email) dai header specificati."""
    target_names = {name.lower() for name in header_names}
    for key, value in headers.items():
        if key.lower() in target_names and value:
            for item in getaddresses([value]):
                name, email = item
                if email:
                    yield name, email


def normalize_phone(phone: str) -> Optional[str]:
    """Normalizza numero di telefono."""
    # Rimuovi tutto tranne cifre e +
    digits = re.sub(r"[^\d+]", "", phone)
    
    # Conta cifre effettive (escluso +)
    digit_count = len(re.sub(r"\D", "", digits))
    if digit_count < 7:
        return None
    
    # Converti 00 iniziale in +
    if digits.startswith("00"):
        digits = "+" + digits[2:]
    
    # Standardizza prefisso italiano
    if digits.startswith("39") and not digits.startswith("+"):
        digits = "+39" + digits[2:]
    elif re.match(r'^3\d{8,9}$', digits):  # cellulare IT senza prefisso
        digits = "+39" + digits
    
    return digits


def extract_org_from_line(line: str) -> Optional[str]:
    """Estrae organizzazione da una riga con pattern contestuali."""
    stripped = line.strip()
    if not stripped:
        return None
    
    low = stripped.lower()
    
    # Pattern: "Azienda: Nome S.r.l."
    if ":" in stripped:
        before, after = stripped.split(":", 1)
        if any(h in before.lower() for h in ORG_HINTS):
            candidate = after.strip()
            if candidate:
                return candidate
    
    # Pattern: "Azienda - Nome S.r.l."
    if "-" in stripped:
        parts = stripped.split("-", 1)
        before, after = parts[0].strip(), parts[1].strip()
        if any(h in before.lower() for h in ORG_HINTS):
            if after:
                return after
    
    # Pattern: "Nome S.r.l." (riga contiene sigla societaria)
    societale_pattern = r'\b(?:s\.r\.l|srl|s\.p\.a|spa|inc|ltd|llc|corp|gmbh|sarl)\b'
    if re.search(societale_pattern, low):
        # Rimuovi prefissi comuni
        candidate = re.sub(
            r"(?i)^(?:azienda|company|impresa|organizzazione|org|società|societa)[:\-\s]*",
            "",
            stripped,
        ).strip(" -")
        if candidate:
            return candidate
    
    # Fallback: riga con hint ma senza separatore chiaro
    if any(h in low for h in ORG_HINTS):
        candidate = re.sub(
            r"(?i)^(?:azienda|company|impresa|organizzazione|org|società|societa)[:\-\s]*",
            "",
            stripped,
        ).strip(" -")
        if candidate and len(candidate) > 3:
            return candidate
    
    return None


def parse_contact_fields(
    text: str, 
    headers: dict[str, str] | None = None
) -> Dict[str, Optional[str]]:
    """
    Parser euristico migliorato per estrarre contatti da email.
    
    Priorità:
    1. Email: Reply-To > From > primo match in body
    2. Nome: Reply-To > From
    3. Telefono: label-based scoring (preferisce cell/mobile)
    4. Org: pattern contestuali con sigla societaria
    """
    headers = headers or {}
    result = {
        "first_name": None,
        "last_name": None,
        "email": None,
        "phone": None,
        "org": None,
    }

    # === EMAIL & NOME da header ===
    
    # Priorità 1: Reply-To
    for name, email in get_header_addresses(headers, ["reply-to"]):
        if email:
            result["email"] = email.strip()
        if name:
            set_result_name(result, name, overwrite=True)
        break  # prendi solo il primo
    
    # Priorità 2: From (se Reply-To non presente)
    if not result["email"]:
        for name, email in get_header_addresses(headers, ["from"]):
            if email:
                result["email"] = email.strip()
            if name:
                set_result_name(result, name)
            break
    
    # Fallback: email nel body
    if not result["email"]:
        match = EMAIL_RE.search(text)
        if match:
            result["email"] = match.group(0)

    # === TELEFONO con scoring contestuale ===

    phone_candidates: list[Tuple[int, int, str]] = []  # (score, idx, number)
    seen_numbers: set[str] = set()

    # === ORGANIZZAZIONE da frasi esplicite ===
    _try_extract_company_from_text(text, result)

    for idx, line in enumerate(text.splitlines()):
        matches = list(PHONE_RE.finditer(line))
        if not matches:
            continue

        low = line.lower()
        
        # Skip righe con solo fax
        if "fax" in low and not any(h in low for h in ("cell", "mobile", "tel")):
            continue
        
        # Calcola score in base a label
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
    
    # Fallback: scan intero testo se niente trovato in righe labellate
    if not phone_candidates:
        for m in PHONE_RE.finditer(text):
            normalized = normalize_phone(m.group(0))
            if normalized and normalized not in seen_numbers:
                seen_numbers.add(normalized)
                phone_candidates.append((0, len(phone_candidates), normalized))
    
    # Seleziona telefono con score più alto
    if phone_candidates:
        phone_candidates.sort(key=lambda x: (-x[0], x[1]))  # score desc, poi ordine
        result["phone"] = phone_candidates[0][2]

    # === ORGANIZZAZIONE ===
    
    if not result.get("org"):
        for line in text.splitlines():
            org = extract_org_from_line(line)
            if org:
                result["org"] = org
                break

    return result