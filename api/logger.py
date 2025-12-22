import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_NAME = "ocr"
LOG_LEVEL = logging.DEBUG
MAX_BYTES = 5_000_000
BACKUP_COUNT = 5


def get_logger(name: str = LOG_NAME) -> logging.Logger:
    """
    Returns an isolated logger with file + console handlers.
    Safe with Uvicorn / FastAPI.
    """

    logger = logging.getLogger(name)

    # 🔒 CRITICAL: isolate from uvicorn root logger
    logger.propagate = False
    logger.setLevel(LOG_LEVEL)

    # Prevent duplicate handlers (reload-safe)
    if getattr(logger, "_configured", False):
        return logger

    # 📁 logs directory (absolute, stable)
    base_dir = Path(__file__).resolve().parents[1]  # project root
    print(f"Base dir for logs: {base_dir}")
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "api_ocr.log"

    # 🗂️ File handler (rotating)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)

    # 🖥️ Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Mark as configured
    logger._configured = True

    logger.debug("Logger initialized")
    return logger
