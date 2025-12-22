# =========================================
# Imports & Types
# =========================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any

# =========================================
# Type aliases
# =========================================
ValidatorFn = Callable[[str, dict[str, Any]], tuple[bool, list[str]]]
NormalizerFn = Callable[[str], str]
PostCheckFn = Callable[[str], tuple[bool, list[str]]]

# =========================================
# Rule
# =========================================
@dataclass(frozen=True)
class FieldRule:
    threshold: float = 0.8
    normalize: NormalizerFn | None = None
    validate: ValidatorFn | None = None
    post_check: PostCheckFn | None = None

# =========================================
# Registry
# =========================================
class FieldValidatorRegistry:
    def __init__(self) -> None:
        self._rules: dict[str, dict[str, FieldRule]] = {}

    def register(self, document_class: str, label: str, rule: FieldRule) -> None:
        self._rules.setdefault(document_class, {})[label] = rule

    def get(self, document_class: str, label: str) -> FieldRule | None:
        return self._rules.get(document_class, {}).get(label)

    def evaluate(
        self,
        document_class: str,
        label: str,
        raw_value: str,
        score: float,
        ctx: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Retourne: dict prêt à mapper vers FieldResult
        status: OK | LOW_CONFIDENCE | INVALID
        """
        ctx = ctx or {}
        reasons: list[str] = []
        rule = self.get(document_class, label)

        # Pas de règle -> on passe "tel quel"
        if rule is None:
            return {
                "label": label,
                "raw": raw_value,
                "value": raw_value,
                "score": score,
                "status": "OK" if score >= 0.0 else "INVALID",
                "reasons": [],
                "extra": {"rule": "none"},
            }

        # 1) threshold
        status = "OK"
        if score < rule.threshold:
            status = "LOW_CONFIDENCE"
            reasons.append(f"score<{rule.threshold}")

        # 2) normalize
        value = raw_value
        if rule.normalize:
            try:
                value = rule.normalize(raw_value)
            except Exception as e:
                status = "INVALID"
                reasons.append(f"normalize_error:{type(e).__name__}")

        # 3) validate (métier, dépend du contexte)
        if status != "INVALID" and rule.validate:
            ok, rs = rule.validate(value, ctx)
            if not ok:
                status = "INVALID"
                reasons.extend(rs)

        # 4) post_check (regex/checksum)
        if status != "INVALID" and rule.post_check:
            ok, rs = rule.post_check(value)
            if not ok:
                status = "INVALID"
                reasons.extend(rs)

        return {
            "label": label,
            "raw": raw_value,
            "value": value if status != "INVALID" else None,
            "score": score,
            "status": status,
            "reasons": reasons,
            "extra": {"threshold": rule.threshold},
        }