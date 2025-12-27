# =========================================
# Imports
# =========================================
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from api.config import PRINT_TYPE_ONNX

from api.logger import get_logger
logger = get_logger("PRINT TYPE")

# =====================================================
# CONFIG
# =====================================================
IMG_SIZE = (64, 256)  # (H, W)
CLASSES = ["handwritten", "printed"]

# =====================================================
# ONNX Runtime session (loaded ONCE)
# =====================================================
_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
_ORT_SESSION = ort.InferenceSession(
    str(PRINT_TYPE_ONNX),
    providers=_PROVIDERS,
)
_INPUT_NAME = _ORT_SESSION.get_inputs()[0].name

# =====================================================
# Preprocess
# =====================================================
def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))  # C, H, W
    return img[None, ...]


# =====================================================
# Public API
# =====================================================
def classify_print_type(
    crop_bgr: np.ndarray,
) -> tuple[str, float]:
    """
    Returns:
        (print_type, confidence)
    """
    x = _preprocess(crop_bgr)
    logits = _ORT_SESSION.run(None, {_INPUT_NAME: x})[0]

    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    idx = int(probs.argmax(axis=1)[0])
    return CLASSES[idx], float(probs[0, idx])