# =========================================
# Imports
# =========================================
import re
from datetime import datetime, timedelta
from stdnum import iban

# =========================================
# REGEX
# =========================================
NIR_REGEX = re.compile(r"^\s*[12]\d{2}(0[1-9]|1[0-2])\d{2}\d{3}\d{3}\d{2}\s*$")
IBAN_SIMPLE = re.compile(r"\b[A-Z]{2}\s?\d{2}(?:\s?[A-Z0-9]){11,30}\b")
BIC_REGEX = re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b")

# =========================================
# Normalizers
# =========================================
def norm_spaces(v: str) -> str:
    return " ".join(v.split())

def norm_amount(v: str) -> str:
    v = v.replace("€", "").replace("\u00a0", " ").strip()
    v = v.replace(",", ".")
    v = norm_spaces(v)
    # garde chiffres + point
    v = re.sub(r"[^0-9.]", "", v)
    return v

def norm_date(v: str) -> str:
    v = norm_spaces(v).replace(".", "/").replace("-", "/")
    return v

def norm_nir(v: str) -> str:
    return re.sub(r"\s+", "", v)

def norm_iban(v: str) -> str:
    return re.sub(r"\s+", "", v).upper()

def norm_bic(v: str) -> str:
    return re.sub(r"\s+", "", v).upper()

# =========================================
# Validators (business logic)
# =========================================
def validate_amount_range(max_eur: float):
    def _v(value: str, ctx: dict) -> tuple[bool, list[str]]:
        try:
            f = float(value)
            if f <= 0:
                return False, ["amount<=0"]
            if f >= max_eur:
                return False, [f"amount>={max_eur}"]
            return True, []
        except Exception:
            return False, ["amount_not_number"]
    return _v

def validate_date_not_future(max_years_past: int = 3):
    def _v(value: str, ctx: dict) -> tuple[bool, list[str]]:
        # parsing simple JJ/MM/AAAA ou JJ/MM/AA
        m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", value)
        if not m:
            return False, ["date_format"]
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        try:
            dt = datetime(y, mo, d)
        except Exception:
            return False, ["date_invalid"]
        now = datetime.now()
        if dt > now:
            return False, ["date_in_future"]
        if dt < now - timedelta(days=365 * max_years_past):
            return False, ["date_too_old"]
        return True, []
    return _v

def validate_person_name(value: str, ctx: dict) -> tuple[bool, list[str]]:
    v = norm_spaces(value)
    if len(v.split()) < 2:
        return False, ["name_too_short"]
    if any(c.isdigit() for c in v):
        return False, ["name_has_digits"]
    return True, []

def validate_patient_not_same_as_person(value: str, ctx: dict) -> tuple[bool, list[str]]:
    ok, rs = validate_person_name(value, ctx)
    if not ok:
        return ok, rs
    person = (ctx.get("person") or "").strip().lower()
    if person and value.strip().lower() == person:
        return False, ["patient_equals_person"]
    return True, []

# =========================================
# Post-checks
# =========================================
def check_nir(v: str) -> tuple[bool, list[str]]:
    vv = norm_nir(v)
    if not NIR_REGEX.match(vv):
        return False, ["nir_regex"]
    return True, []

def check_iban(v: str) -> tuple[bool, list[str]]:
    vv = norm_iban(v)
    if not IBAN_SIMPLE.match(vv):
        return False, ["iban_regex"]
    # checksum “light” (optionnel: stdnum.iban.is_valid)
    # Si tu as python-stdnum, remplace par iban.is_valid(vv)
    return True, []

def check_bic(v: str) -> tuple[bool, list[str]]:
    vv = norm_bic(v)
    if not BIC_REGEX.match(vv):
        return False, ["bic_regex"]
    return True, []
