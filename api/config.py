# =========================================
# Imports
# =========================================
import os
import logging
from pathlib import Path

LOGGER_NAME = "ocr-api"
logger = logging.getLogger(LOGGER_NAME)

# =========================================
# API Metadata
# =========================================
PROJECT_NAME: str = "docTR API template"
PROJECT_DESCRIPTION: str = "Template API for Optical Character Recognition"
VERSION: str = "0.0.1"
DEBUG: bool = os.environ.get("DEBUG", "false").lower() in ("1", "true", "yes")
LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO
SAVE_OCR_BOXES: bool = os.environ.get("SAVE_OCR_BOXES", "false").lower() in ("1", "true", "yes") or DEBUG

# =========================================
# SAVE IMG Cache directory
# =========================================
OCR_BOXES_DIR = os.environ.get(
    "OCR_BOXES_DIR",
    str(Path(__file__).resolve().parents[1] / "logs" / "ocr_boxes"),
)

# =========================================
# ONNXTR Cache directory
# =========================================
ONNXTR_CACHE_DIR = os.environ.get(
"ONNXTR_CACHE_DIR",
    str(Path(__file__).resolve().parents[1] / "models" / "doctr")
)

os.environ.setdefault("ONNXTR_CACHE_DIR", str(ONNXTR_CACHE_DIR))

# =========================================
# PRINT TYPE Cache directory
# =========================================
PRINT_TYPE_ONNX = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "print_type"
    / "printed_handwritten_resnet18.onnx"
)

# =========================================
# GLINER Cache directory
# =========================================
GLINER_CACHE_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "gliner"
    / "gliner_large-v2.5"
)


logger.info("Configuration loaded")
logger.debug(f"DEBUG={DEBUG}")