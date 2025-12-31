# =========================================
# Imports
# =========================================
import re
from api.logger import get_logger
from datetime import datetime, timedelta
from typing import Any
from stdnum import iban as std_iban
from stdnum.fr import nir as std_nir
from stdnum.fr import siren as std_siren
from stdnum.fr import siret as std_siret
from api.ner.patterns import NIR_REGEX, IBAN_REGEX, BIC_REGEX

# =========================================
# Logger
# =========================================
logger = get_logger(__name__)

# =========================================
# Normalizers
# =========================================
def norm_spaces(v: str) -> str:
    """Normaliser les espaces multiples en espaces uniques."""
    if not isinstance(v, str):
        logger.warning(f"norm_spaces: expected str, got {type(v).__name__}")
        v = str(v)
    if not v:
        return ""
    return " ".join(v.split())

def norm_amount(v: str) -> str:
    """Normaliser un montant financier."""
    if not isinstance(v, str):
        logger.warning(f"norm_amount: expected str, got {type(v).__name__}")
        v = str(v)
    if not v or not v.strip():
        return ""
    try:
        v = v.replace("€", "").replace("\u00a0", " ").strip()
        v = v.replace(",", ".")
        v = norm_spaces(v)
        # garde chiffres + point
        v = re.sub(r"[^0-9.]", "", v)
        return v
    except Exception as e:
        logger.error(f"norm_amount: erreur lors du traitement '{v}': {e}")
        return ""

def norm_date(v: str) -> str:
    """Normaliser une date."""
    if not isinstance(v, str):
        logger.warning(f"norm_date: expected str, got {type(v).__name__}")
        v = str(v)
    if not v or not v.strip():
        return ""
    try:
        v = norm_spaces(v)
        # Corrections OCR courantes + uniformiser les séparateurs
        v = v.replace("\\", "/").replace(".", "/").replace("-", "/")
        v = v.replace("O", "0").replace("o", "0")
        v = v.replace("I", "1").replace("l", "1")
        # Garder uniquement chiffres et séparateurs
        v = re.sub(r"[^0-9/]", "", v)
        v = re.sub(r"/{2,}", "/", v).strip("/")
        # Normaliser si on trouve une date lisible
        m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", v)
        if not m:
            return v
        d, mo, y = m.group(1), m.group(2), m.group(3)
        if len(y) == 2:
            y = f"20{y}"
        return f"{int(d):02d}/{int(mo):02d}/{y}"
    except Exception as e:
        logger.error(f"norm_date: erreur lors du traitement '{v}': {e}")
        return ""

def norm_alnum(v: str) -> str:
    """Normaliser en alphanumériques majuscules uniquement."""
    if not isinstance(v, str):
        logger.warning(f"norm_alnum: expected str, got {type(v).__name__}")
        v = str(v)
    if not v or not v.strip():
        return ""
    return re.sub(r"[^0-9A-Z]", "", v.upper())

def norm_nir(v: str) -> str:
    """Normaliser un NIR."""
    if not isinstance(v, str):
        logger.warning(f"norm_nir: expected str, got {type(v).__name__}")
        v = str(v)
    if not v or not v.strip():
        return ""
    return norm_alnum(v)

def norm_iban(v: str) -> str:
    """Normaliser un IBAN."""
    if not isinstance(v, str):
        logger.warning(f"norm_iban: expected str, got {type(v).__name__}")
        v = str(v)
    if not v or not v.strip():
        return ""
    return norm_alnum(v)

def norm_bic(v: str) -> str:
    """Normaliser un BIC."""
    if not isinstance(v, str):
        logger.warning(f"norm_bic: expected str, got {type(v).__name__}")
        v = str(v)
    if not v or not v.strip():
        return ""
    return re.sub(r"\s+", "", v).upper()

# =========================================
# Validators (business logic)
# =========================================
def validate_amount_range(max_eur: float):
    """Valider qu'un montant est dans la plage autorisée."""
    # Validation du paramètre
    if max_eur is None or not isinstance(max_eur, (int, float)):
        logger.error(f"validate_amount_range: max_eur doit être numérique, reçu {type(max_eur).__name__}")
        raise ValueError(f"max_eur doit être un nombre, reçu {type(max_eur).__name__}")
    if max_eur <= 0:
        logger.error(f"validate_amount_range: max_eur doit être > 0, reçu {max_eur}")
        raise ValueError(f"max_eur doit être > 0")
    
    def _v(value: str, ctx: dict) -> tuple[bool, list[str]]:
        # Validation des paramètres
        if value is None:
            return False, ["amount_null"]
        if not isinstance(value, str):
            logger.warning(f"validate_amount_range: valeur attendue str, reçu {type(value).__name__}")
            value = str(value)
        if not value or not value.strip():
            return False, ["amount_empty"]
        if ctx is None:
            logger.warning("validate_amount_range: ctx est None")
            ctx = {}
        
        try:
            f = float(value)
            if f <= 0:
                return False, ["amount_must_be_positive"]
            if f >= max_eur:
                return False, [f"amount_exceeds_max_{max_eur}"]
            return True, []
        except ValueError as e:
            logger.debug(f"validate_amount_range: impossible de convertir '{value}' en float: {e}")
            return False, ["amount_not_valid_number"]
        except Exception as e:
            logger.error(f"validate_amount_range: erreur inattendue pour '{value}': {e}")
            return False, ["amount_validation_error"]
    return _v

def validate_date_not_future(max_years_past: int = 3):
    """Valider qu'une date n'est pas dans le futur et pas trop ancienne."""
    # Validation du paramètre
    if max_years_past is None or not isinstance(max_years_past, int):
        logger.error(f"validate_date_not_future: max_years_past doit être int, reçu {type(max_years_past).__name__}")
        raise ValueError(f"max_years_past doit être un entier")
    if max_years_past < 0:
        logger.error(f"validate_date_not_future: max_years_past ne peut pas être négatif")
        raise ValueError(f"max_years_past ne peut pas être négatif")
    
    def _v(value: str, ctx: dict) -> tuple[bool, list[str]]:
        # Validation des paramètres
        if value is None:
            return False, ["date_null"]
        if not isinstance(value, str):
            logger.warning(f"validate_date_not_future: valeur attendue str, reçu {type(value).__name__}")
            value = str(value)
        if not value or not value.strip():
            return False, ["date_empty"]
        if ctx is None:
            logger.warning("validate_date_not_future: ctx est None")
            ctx = {}
        
        try:
            # Parsing simple JJ/MM/AAAA ou JJ/MM/AA
            m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", value)
            if not m:
                return False, ["date_format_invalid"]
            
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            
            # Validation des plages
            if not (1 <= mo <= 12):
                return False, ["date_month_invalid"]
            if not (1 <= d <= 31):
                return False, ["date_day_invalid"]
            
            if y < 100:
                y += 2000
            
            try:
                dt = datetime(y, mo, d)
            except ValueError as e:
                logger.debug(f"validate_date_not_future: date invalide {d}/{mo}/{y}: {e}")
                return False, ["date_invalid"]
            
            now = datetime.now()
            if dt > now:
                return False, ["date_in_future"]
            if dt < now - timedelta(days=365 * max_years_past):
                return False, [f"date_too_old_max_{max_years_past}_years"]
            return True, []
        except Exception as e:
            logger.error(f"validate_date_not_future: erreur inattendue pour '{value}': {e}")
            return False, ["date_validation_error"]
    return _v

def validate_person_name(value: str, ctx: dict) -> tuple[bool, list[str]]:
    """Valider un nom de personne (au moins 2 mots, pas de chiffres)."""
    # Validation des paramètres
    if value is None:
        return False, ["name_null"]
    if not isinstance(value, str):
        logger.warning(f"validate_person_name: valeur attendue str, reçu {type(value).__name__}")
        value = str(value)
    if not value or not value.strip():
        return False, ["name_empty"]
    if ctx is None:
        logger.warning("validate_person_name: ctx est None")
        ctx = {}
    
    try:
        v = norm_spaces(value)
        
        # Vérifications
        words = v.split()
        if len(words) < 2:
            return False, ["name_too_short"]
        if any(c.isdigit() for c in v):
            return False, ["name_contains_digits"]
        if len(v) < 3:
            return False, ["name_too_short"]
        if len(v) > 100:
            return False, ["name_too_long"]
        
        return True, []
    except Exception as e:
        logger.error(f"validate_person_name: erreur inattendue pour '{value}': {e}")
        return False, ["name_validation_error"]

def validate_patient_not_same_as_person(value: str, ctx: dict) -> tuple[bool, list[str]]:
    """Valider qu'un nom de patient n'est pas identique à la personne."""
    # Validation des paramètres
    if value is None:
        return False, ["patient_null"]
    if not isinstance(value, str):
        logger.warning(f"validate_patient_not_same_as_person: valeur attendue str, reçu {type(value).__name__}")
        value = str(value)
    if not value or not value.strip():
        return False, ["patient_empty"]
    if ctx is None:
        logger.warning("validate_patient_not_same_as_person: ctx est None")
        ctx = {}
    
    try:
        # Vérifier d'abord que c'est un nom valide
        ok, rs = validate_person_name(value, ctx)
        if not ok:
            return ok, rs
        
        # Comparer avec le champ "person" du contexte
        person = (ctx.get("person") or "").strip().lower()
        patient_lower = value.strip().lower()
        
        if person and patient_lower == person:
            return False, ["patient_equals_person"]
        
        return True, []
    except Exception as e:
        logger.error(f"validate_patient_not_same_as_person: erreur inattendue pour '{value}': {e}")
        return False, ["patient_validation_error"]

# =========================================
# Post-checks
# =========================================
def check_nir(v: str) -> tuple[bool, list[str]]:
    """Valider un NIR avec regex et checksum."""
    if v is None:
        return False, ["nir_null"]
    if not isinstance(v, str):
        logger.warning(f"check_nir: valeur attendue str, reçu {type(v).__name__}")
        v = str(v)
    if not v or not v.strip():
        return False, ["nir_empty"]
    
    try:
        vv = norm_nir(v)
        if not vv:
            return False, ["nir_normalization_failed"]
        
        # Vérifier la longueur
        if len(vv) != 15:
            return False, ["nir_invalid_length"]
        
        # Vérifier le format regex
        if not NIR_REGEX.match(vv):
            return False, ["nir_regex_format_invalid"]
        
        # Vérifier le checksum
        if not std_nir.is_valid(vv):
            return False, ["nir_checksum_invalid"]
        
        return True, []
    except Exception as e:
        logger.error(f"check_nir: erreur inattendue pour '{v}': {e}")
        return False, ["nir_validation_error"]

def check_iban(v: str) -> tuple[bool, list[str]]:
    """Valider un IBAN avec regex et checksum."""
    if v is None:
        return False, ["iban_null"]
    if not isinstance(v, str):
        logger.warning(f"check_iban: valeur attendue str, reçu {type(v).__name__}")
        v = str(v)
    if not v or not v.strip():
        return False, ["iban_empty"]
    
    try:
        vv = norm_iban(v)
        if not vv:
            return False, ["iban_normalization_failed"]
        
        # Vérifier la longueur (IBAN: 15-34 caractères)
        if len(vv) < 15 or len(vv) > 34:
            return False, ["iban_invalid_length"]
        
        # Vérifier le format regex
        if not IBAN_REGEX.match(vv):
            return False, ["iban_regex_format_invalid"]
        
        # Vérifier le checksum
        if not std_iban.is_valid(vv):
            return False, ["iban_checksum_invalid"]
        
        return True, []
    except Exception as e:
        logger.error(f"check_iban: erreur inattendue pour '{v}': {e}")
        return False, ["iban_validation_error"]

def check_bic(v: str) -> tuple[bool, list[str]]:
    """Valider un code BIC."""
    if v is None:
        return False, ["bic_null"]
    if not isinstance(v, str):
        logger.warning(f"check_bic: valeur attendue str, reçu {type(v).__name__}")
        v = str(v)
    if not v or not v.strip():
        return False, ["bic_empty"]
    
    try:
        vv = norm_bic(v)
        if not vv:
            return False, ["bic_normalization_failed"]
        
        # Vérifier la longueur (BIC: 8 ou 11 caractères)
        if len(vv) not in (8, 11):
            return False, ["bic_invalid_length"]
        
        # Vérifier le format regex
        if not BIC_REGEX.match(vv):
            return False, ["bic_regex_format_invalid"]
        
        return True, []
    except Exception as e:
        logger.error(f"check_bic: erreur inattendue pour '{v}': {e}")
        return False, ["bic_validation_error"]
