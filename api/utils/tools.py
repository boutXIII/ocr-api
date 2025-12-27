# =========================================
# Imports
# =========================================
import os
import json
import base64
import importlib
from typing import Any, Optional
from pathlib import Path

import cv2
import numpy as np
from fastapi import UploadFile
from onnxtr.io import DocumentFile
from onnxtr.models import EngineConfig, ocr_predictor
from onnxtr.models.predictor import OCRPredictor

from gliner import GLiNER

from api.config import GLINER_CACHE_DIR, ONNXTR_CACHE_DIR
from api.logger import get_logger
from api.vision.model_store import ModelStore
logger = get_logger("TOOLS_UTILS")

# =========================================
# Geometry helpers
# =========================================
def resolve_geometry(
    geom: Any,
) -> tuple[float, float, float, float] | tuple[float, float, float, float, float, float, float, float]:
    if len(geom) == 4:
        return (*geom[0], *geom[1], *geom[2], *geom[3])
    return (*geom[0], *geom[1])

# =========================================
# Document loading (FastAPI input)
# =========================================
async def get_documents(
    request,
    file: Optional[UploadFile] = None,
) -> tuple[DocumentFile, str]:
    """Convert a list of UploadFile objects to lists of numpy arrays and their corresponding filenames
    Support:
    - UploadFile (multipart)
    - body brut (image/pdf)
    - JSON base64

    Args:
        request: Request object containing the files to be processed

    Returns:
    - DocumentFile
    - filename

    """

    filename = "document"

    if file is not None:
        content = await file.read()
        filename = file.filename or filename

    else:
        content = await request.body()

        if not content:
            raise ValueError("Empty body")

        # Tentative JSON base64
        try:
            payload = json.loads(content)
            if isinstance(payload, dict) and "fileBase64" in payload:
                content = base64.b64decode(payload["fileBase64"])
                filename = payload.get("filename", filename)
        except Exception:
            pass  # body binaire brut

    try:
        if (
            filename.lower().endswith(".pdf")
            or content[:4] == b"%PDF"
        ):
            doc = DocumentFile.from_pdf(content)
        else:
            doc = DocumentFile.from_images(content)
    except Exception as e:
        raise ValueError(f"Error loading document: {e}")

    return doc, filename

# =========================================
# ONNX model resolution / loading
# =========================================
def find_model_path(model_name: str, models_dir: str = ONNXTR_CACHE_DIR) -> str:
    """
    Recherche un fichier .pt correspondant au modèle choisi dans le cache local.
    Exemple : det_arch='db_resnet50' → 'db_resnet50-xxxx.pt'
    """
    # logger.debug("find_model_path")
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Le dossier '{models_dir}' n'existe pas.")
    
    for file in os.listdir(models_dir):
        if file.startswith(model_name) and file.endswith(".onnx"):
            return os.path.join(models_dir, file)
    raise FileNotFoundError(f"Aucun fichier trouvé pour {model_name} dans {models_dir}")

def load_det_models(det_arch: str = "db_resnet50"):
    """Load detection model
    
    Args:
        det_arch: Detection architecture name
        
    Returns:
        Detection model
    """
    print(f"Loading detection model: {det_arch}...")
    det_path = find_model_path(det_arch)
    det_module = importlib.import_module("onnxtr.models.detection")
    det_fn = getattr(det_module, det_arch)
    det_model = det_fn(det_path)
    logger.debug(f"Detection model loaded: {det_arch}")
    logger.debug(f"Detection model loaded from {det_path}")
    logger.debug(f"Detection model function: {det_fn}")
    logger.debug(f"Detection model details: {det_model}")
    
    return det_model
        
def load_reco_models(reco_arch: str = "crnn_vgg16_bn"):
    """Load recognition model
    
    Args:
        reco_arch: Recognition architecture name
        
    Returns:
        Recognition model
    """
    print(f"Loading recognition model: {reco_arch}...")
    reco_path = find_model_path(reco_arch)
    reco_module = importlib.import_module("onnxtr.models.recognition")
    reco_fn = getattr(reco_module, reco_arch)
    reco_model = reco_fn(reco_path)
    logger.debug(f"Recognition model loaded: {reco_arch}")
    logger.debug(f"Recognition model loaded from {reco_path}")
    logger.debug(f"Recognition model function: {reco_fn}")
    logger.debug(f"Recognition model details: {reco_model}")

    return reco_model

def load_page_orientation_models(orientation_arch: str = "mobilenet_v3_small_page_orientation"):
    """Load page orientation model

    Args:
        reco_arch: Recognition architecture name
        
    Returns:
        Orientation model
    """
    print(f"Loading page orientation model: {orientation_arch}...")
    orientation_path = find_model_path(orientation_arch)
    orientation_module = importlib.import_module("onnxtr.models.classification")
    orientation_fn = getattr(orientation_module, orientation_arch)
    orientation_model = orientation_fn(orientation_path)

    logger.debug(f"Orientation model loaded: {orientation_arch}")
    logger.debug(f"Orientation model loaded from {orientation_path}")
    logger.debug(f"Orientation model function: {orientation_fn}")
    logger.debug(f"Orientation model details: {orientation_model}")

    zoo_module = importlib.import_module("onnxtr.models.classification.zoo")
    orientation_predictor = zoo_module.page_orientation_predictor(orientation_model)

    return orientation_predictor

def load_crop_orientation_models(orientation_arch: str = "mobilenet_v3_small_crop_orientation"):
    """Load crop orientation model

    Args:
        reco_arch: Recognition architecture name
        
    Returns:
        Crop orientation model
    """
    print(f"Loading crop orientation model: {orientation_arch}...")
    orientation_path = find_model_path(orientation_arch)
    orientation_module = importlib.import_module("onnxtr.models.classification")
    orientation_fn = getattr(orientation_module, orientation_arch)
    orientation_model = orientation_fn(orientation_path)

    logger.debug(f"Orientation model loaded: {orientation_arch}")
    logger.debug(f"Orientation model loaded from {orientation_path}")
    logger.debug(f"Orientation model function: {orientation_fn}")
    logger.debug(f"Orientation model details: {orientation_model}")

    zoo_module = importlib.import_module("onnxtr.models.classification.zoo")
    orientation_predictor = zoo_module.crop_orientation_predictor(orientation_model)

    return orientation_predictor

def load_predictor(
    det_arch: str,
    reco_arch: str,
    detect_language: bool,
    detect_orientation: bool,
    bin_thresh: float,
    box_thresh: float,
) -> OCRPredictor:
    """Charge un predictor docTR en ONNX via OnnxTR.

    Signature compatible avec backend/ocr/doctr/pytorch.py pour
    pouvoir simplement changer l'import dans le reste du projet.

    Args:
        det_arch: architecture de détection (nom string)
        reco_arch: architecture de reconnaissance (nom string)
        assume_straight_pages: supposer les pages droites
        straighten_pages: redresser les pages
        export_as_straight_boxes: exporter des boîtes droites
        disable_page_orientation: désactiver la détection d’orientation de page
        disable_crop_orientation: désactiver la détection d’orientation de crop
        detect_language: activer la détection de langue
        bin_thresh: seuil de binarisation de la carte de segmentation
        box_thresh: seuil minimal de détection de boîte

    Returns:
        OCRPredictor (OnnxTR)
    """
    logger.debug("load_predictor")
    # Config ONNXRuntime (CPU)
    engine_cfg = EngineConfig(
        providers=[
            ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})
        ]
    )

    # Chargement des modèles
    det_model = load_det_models(det_arch)
    reco_model = load_reco_models(reco_arch)

    if detect_orientation:
        logger.info("Orientation detection enabled")

        predictor = ocr_predictor(
            det_arch=det_model,
            reco_arch=reco_model,
            assume_straight_pages=False,
            straighten_pages=True,
            export_as_straight_boxes=True,
            detect_orientation=detect_orientation,
            disable_page_orientation=True,
            disable_crop_orientation=True,
            detect_language=detect_language,
            det_engine_cfg=engine_cfg,
            reco_engine_cfg=engine_cfg,
            clf_engine_cfg=engine_cfg,
        )

        # Applique les seuils de post-processing comme dans la version PyTorch
        predictor.det_predictor.model.postprocessor.bin_thresh = bin_thresh
        predictor.det_predictor.model.postprocessor.box_thresh = box_thresh

        predictor.page_orientation_predictor = load_page_orientation_models()
        predictor.crop_orientation_predictor = load_crop_orientation_models()

        predictor._page_orientation_disabled = False
        predictor._page_orientation_disabled = False
    else:
        logger.info("Orientation detection disabled")

        predictor = ocr_predictor(
            det_arch=det_model,
            reco_arch=reco_model,
            assume_straight_pages=True,
            straighten_pages=False,
            export_as_straight_boxes=False,
            detect_orientation=detect_orientation,
            disable_page_orientation=True,
            disable_crop_orientation=True,
            detect_language=detect_language,
            det_engine_cfg=engine_cfg,
            reco_engine_cfg=engine_cfg,
            clf_engine_cfg=engine_cfg,
        )

        # Applique les seuils de post-processing comme dans la version PyTorch
        predictor.det_predictor.model.postprocessor.bin_thresh = bin_thresh
        predictor.det_predictor.model.postprocessor.box_thresh = box_thresh

    return predictor

# =========================================
# GLiNER (NER)
# =========================================
def load_gliner_model() -> GLiNER:
    """Load GLiNER model from local path

    Returns:
        GLiNER model
    """
    logger.info("Loading GLiNER model...")
    gliner_model = GLiNER.from_pretrained(
        GLINER_CACHE_DIR,
        load_onnx_model=True,
        map_location="cpu",
    )
    logger.info("GLiNER model loaded successfully")
    return gliner_model

def run_gliner(
    *,
    text: str,
    labels: list[str],
    threshold: float,
) -> list[dict]:
    """
    GLiNER engine — no document knowledge here
    """
    gliner_model: GLiNER = ModelStore.get("ner_registry")
    return gliner_model.predict_entities(
        text=text,
        labels=labels,
        threshold=threshold,
    )
