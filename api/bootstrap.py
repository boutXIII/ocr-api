# =========================================
# Imports
# =========================================
from api.logger import get_logger
from api.vision.model_store import ModelStore
from api.utils.tools import (
    load_det_models,
    load_gliner_model,
    load_reco_models,
    load_page_orientation_models,
)

logger = get_logger("BOOTSTRAP")

# =========================================
# CACHE ET MODELES PRELOADING
# =========================================
def preload_models():
    logger.info("🔄 Preloading models...")

    # OCR
    ModelStore.set(
        "detector",
        load_det_models("db_resnet50")
    )
    ModelStore.set(
        "reco",
        load_reco_models("crnn_vgg16_bn")
    )

    # PRINT TYPE
    ModelStore.set(
        "reco_printed",
        load_reco_models("crnn_vgg16_bn")
    )
    ModelStore.set(
        "reco_handwritten",
        load_reco_models("parseq")
    )

    # Orientation
    ModelStore.set(
        "page_orientation",
        load_page_orientation_models()
    )

    # NER
    ModelStore.set(
        "ner_registry",
        load_gliner_model()
    )

    logger.info("✅ Models successfully preloaded")
