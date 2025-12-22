# =========================================
# Imports
# =========================================
import re
from typing import Any

from api.ner import rules
from api.ner.registry import FieldRule, FieldValidatorRegistry
from api.ner.patterns import NIR_REGEX, IBAN_REGEX, BIC_REGEX

# =========================================
# Document strategy definition
# =========================================
DOCUMENT_STRATEGY: dict[str, dict[str, dict[str, Any]]] = {
    "FACT_MEDECINE_DOUCE": {
        "person": {
            "extractor": "gliner",
            "rule": FieldRule(
                threshold=0.80,
                normalize=rules.norm_spaces,
                validate=rules.validate_person_name,
            ),
        },
        "patient": {
            "extractor": "gliner",
            "rule": FieldRule(
                threshold=0.80,
                normalize=rules.norm_spaces,
                validate=rules.validate_patient_not_same_as_person,
            ),
        },
        "date": {
            "extractor": "gliner",
            "rule": FieldRule(
                threshold=0.80,
                normalize=rules.norm_date,
                validate=rules.validate_date_not_future(max_years_past=3),
            ),
        },
        "amount": {
            "extractor": "gliner",
            "rule": FieldRule(
                threshold=0.85,
                normalize=rules.norm_amount,
                validate=rules.validate_amount_range(max_eur=1000.0),
            ),
        },
        "speciality": {
            "extractor": "gliner",
            "rule": FieldRule(
                threshold=0.75,
                normalize=rules.norm_spaces,
            ),
        },
        "social_security_number": {
            "extractor": "regex",
            "pattern": NIR_REGEX,
            "score": 1.0,
            "rule": FieldRule(
                threshold=0.90,
                normalize=rules.norm_nir,
                post_check=rules.check_nir,
            ),
        },
    },
    "RIB": {
        "account_holder": {
            "extractor": "gliner",
            "rule": FieldRule(
                threshold=0.80,
                normalize=rules.norm_spaces,
                validate=rules.validate_person_name,
            ),
        },
        "iban_number": {
            "extractor": "regex",
            "pattern": IBAN_REGEX,
            "score": 1.0,
            "rule": FieldRule(
                threshold=0.90,
                normalize=rules.norm_iban,
                post_check=rules.check_iban,
            ),
        },
        "bic_code": {
            "extractor": "gliner",
            "pattern": BIC_REGEX,
            "rule": FieldRule(
                threshold=0.85,
                normalize=rules.norm_bic,
                post_check=rules.check_bic,
            ),
        },
    },
}

# =========================================
# Registry builder
# =========================================
def build_registry() -> FieldValidatorRegistry:
    """
    Construit le FieldValidatorRegistry à partir de DOCUMENT_STRATEGY
    """
    reg = FieldValidatorRegistry()

    for document_class, fields in DOCUMENT_STRATEGY.items():
        for label, cfg in fields.items():
            rule = cfg.get("rule")
            if rule:
                reg.register(document_class, label, rule)

    return reg