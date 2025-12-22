import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(name: str = "ocr"):
    """
    Returns a preconfigured logger usable anywhere in the project.
    """

    # Create logs directory
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "api_ocr.log")

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate logs
    if logger.handlers:
        return logger

    # File handler (rotating)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Log format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
