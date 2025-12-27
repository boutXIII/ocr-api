# =========================================
# Imports
# =========================================
from typing import Any
from api.logger import get_logger

logger = get_logger("MODEL_STORE")

# =========================================
# MODEL STORE
# =========================================
class ModelStore:
    """
    Central registry for all loaded ML models.
    Loaded once at startup, read-only at runtime.
    """
    _models: dict[str, Any] = {}

    @classmethod
    def set(cls, key: str, value: Any):
        logger.debug(f"Registering model: {key}")
        cls._models[key] = value

    @classmethod
    def get(cls, key: str) -> Any:
        if key not in cls._models:
            raise RuntimeError(f"Model '{key}' not loaded")
        return cls._models[key]

    @classmethod
    def has(cls, key: str) -> bool:
        return key in cls._models
