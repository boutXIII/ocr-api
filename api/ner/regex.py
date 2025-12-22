# =========================================
# Imports
# =========================================
import re
from typing import Iterable

# =========================================
# Regex rules
# =========================================
REGEX_RULES = {
    "iban_number": {
        "pattern": re.compile(r"\b[A-Z]{2}\s?\d{2}(?:\s?\d{4}){4,7}\b"),
        "score": 0.9,
    },
    "bic_code": {
        "pattern": re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b"),
        "score": 0.9,
    },
    "bank_account_number": {
        "pattern": re.compile(r"\b\d{11}\b"),
        "score": 0.9,
    },
    "bank_account_key": {
        "pattern": re.compile(r"\b\d{2}\b"),
        "score": 0.9,
    },
    "bank_code": {
        "pattern": re.compile(r"\b\d{5}\b"),
        "score": 0.9,
    },
    "bank_counter_code": {
        "pattern": re.compile(r"\b\d{5}\b"),
        "score": 0.9,
    },
    "social_security_number": {
        "pattern": re.compile(r"\b[12]\d{14}\b"),
        "score": 0.9,
    },
}

# =========================================
# Extraction helper
# =========================================
def extract_regex_entities(text: str, labels: Iterable[str]) -> list[dict]:
    """
    Retourne des entités au même format que GLiNER
    """
    entities: list[dict] = []

    for label in labels:
        rule = REGEX_RULES.get(label)
        if not rule:
            continue

        for m in rule["pattern"].finditer(text):
            entities.append({
                "label": label,
                "text": m.group(),
                "start": m.start(),
                "end": m.end(),
                "score": rule["score"],
            })

    return entities
