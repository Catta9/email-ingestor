from libs.parser import parse_contact_fields

def test_parser_extracts_basic_fields():
    text = """
    Buongiorno,
    sono Mario Rossi dell'azienda: Rossi Impianti S.r.l.
    Telefono: +39 345 678 9012
    Email: mario.rossi@example.com
    """

    headers = {"From": "Mario Rossi <mario.rossi@example.com>"}
    result = parse_contact_fields(text, headers=headers)

    # controlliamo che i campi principali vengano trovati
    assert result["email"] == "mario.rossi@example.com"
    assert result["first_name"] == "Mario"
    assert result["last_name"] == "Rossi"
    assert result["phone"].endswith("3456789012")
    assert "Rossi" in result["org"]

def test_parser_handles_missing_phone():
    text = """
    Salve,
    sono Giulia Bianchi dell'azienda: ACME S.p.A.
    Potete scrivermi a giulia.bianchi@acme.it
    """
    headers = {"From": "Giulia Bianchi <giulia.bianchi@acme.it>"}
    r = parse_contact_fields(text, headers=headers)

    assert r["email"] == "giulia.bianchi@acme.it"
    assert r["first_name"] == "Giulia"
    assert r["last_name"] == "Bianchi"
    assert r["phone"] is None
    assert "ACME" in (r["org"] or "")


def test_parser_company_english_and_email_in_body_only():
    text = """
    Hello,
    I am John Doe, company: Tech Ltd
    Please contact me at john.doe@techltd.co.uk
    """
    headers = {"From": "John Doe <john.doe@techltd.co.uk>"}  # ok anche se uguale al body
    r = parse_contact_fields(text, headers=headers)

    assert r["email"] == "john.doe@techltd.co.uk"
    assert r["first_name"] == "John"
    assert r["last_name"] == "Doe"
    assert "Tech" in (r["org"] or "")


def test_parser_prefers_reply_to_and_structured_signature():
    text = """
    Cordiali saluti,
    Mario Rossi
    Ruolo: Sales Manager
    Tel.: +39 02 1234 5678
    Cell. +39 345 987 6543
    Email: mario.rossi@example.com
    Azienda: Rossi Impianti S.r.l.
    """

    headers = {
        "From": "Customer Care <noreply@service.com>",
        "Reply-To": "Mario Rossi <mario.rossi@example.com>",
    }

    r = parse_contact_fields(text, headers=headers)

    assert r["email"] == "mario.rossi@example.com"
    assert r["first_name"] == "Mario"
    assert r["last_name"] == "Rossi"
    # prefer the cell phone number because of the label
    assert r["phone"].endswith("3459876543")
    assert r["org"] == "Rossi Impianti S.r.l."


def test_parser_skips_fax_and_recognises_other_sigle():
    text = """
    Kind regards,
    Laura Bianchi
    Phone: +33 1 44 55 66 77
    Fax: +33 1 44 55 66 78
    Company - Nouvelle Energie S.A.S.
    """

    headers = {
        "From": "Laura Bianchi <laura.bianchi@example.fr>",
    }

    r = parse_contact_fields(text, headers=headers)

    assert r["email"] == "laura.bianchi@example.fr"
    assert r["phone"].endswith("144556677")
    assert r["org"] == "Nouvelle Energie S.A.S."
