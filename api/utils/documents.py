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

from gliner import GLiNER

# =========================================
# Constantes / Environnement
# =========================================
DEFAULT_ONNXTR_CACHE = "models\doctr"

os.environ.setdefault("ONNXTR_CACHE_DIR", str(DEFAULT_ONNXTR_CACHE))

print(f"ONNXTR_CACHE_DIR set to {os.environ['ONNXTR_CACHE_DIR']}")

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
def find_model_path(model_name: str, models_dir: str = DEFAULT_ONNXTR_CACHE) -> str:
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

def load_reco_models(reco_printed: str = "crnn_vgg16_bn", reco_handwritten: str = "parseq"):
    """Load separate recognition models for printed and handwritten text
    
    Args:
        reco_printed: Recognition model for printed text
        reco_handwritten: Recognition model for handwritten text
        
    Returns:
        Tuple of (reco_printed_model, reco_handwritten_model)
    """
    print(f"Loading printed reco model: {reco_printed}...")
    printed_path = find_model_path(reco_printed)
    reco_module = importlib.import_module("onnxtr.models.recognition")
    reco_printed_fn = getattr(reco_module, reco_printed)
    reco_printed_model = reco_printed_fn(printed_path)
    
    print(f"Loading handwritten reco model: {reco_handwritten}...")
    handwritten_path = find_model_path(reco_handwritten)
    reco_handwritten_fn = getattr(reco_module, reco_handwritten)
    reco_handwritten_model = reco_handwritten_fn(handwritten_path)
    
    return reco_printed_model, reco_handwritten_model

# =========================================
# GLiNER (NER)
# =========================================
gliner_model = GLiNER.from_pretrained(
    "models/gliner/gliner_large-v2.5",
    load_onnx_model=True,
    map_location="cpu",
)

def run_gliner(
    *,
    text: str,
    labels: list[str],
    threshold: float,
) -> list[dict]:
    """
    GLiNER engine — no document knowledge here
    """
    return gliner_model.predict_entities(
        text=text,
        labels=labels,
        threshold=threshold,
    )
