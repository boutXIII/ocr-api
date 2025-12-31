# =========================================
# Imports
# =========================================
import time
from fastapi import (
    APIRouter, Depends, File,
    HTTPException, Request, UploadFile, status
)

from api.logger import get_logger
logger = get_logger("OCR_ROUTE")

from api.schemas import (
    OCRIn, OCROut,
    ReadIn, ReadOut
)
from api.utils.tools import get_documents
from api.vision.predictor import init_predictor

# =========================================
# Router & Registry
# =========================================
router = APIRouter()

# =========================================
# OCR endpoint
# =========================================
@router.post(
        "/",
        response_model=OCROut,
        status_code=status.HTTP_200_OK,
        summary="Perform OCR"
)
async def perform_ocr(
    request: Request,
    ocr_params: OCRIn = Depends(),
    file: UploadFile = File(
        None,
        description="Upload image or PDF file (optional)"
    ),
) -> OCROut:
    """Runs docTR OCR model to analyze the input image"""
    start = time.perf_counter()

    try:
        # generator object to list
        content, filename = await get_documents(request, file)
        logger.debug(f"filenames: {filename}")
        predictor = init_predictor(ocr_params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    results = predictor(content, filename)
    result: OCROut = results[0]

    return result.model_copy(
        update={
            "name": filename,
            "duration": round(time.perf_counter() - start, 4)
        }
    )
    
# =========================================
# OCR + NER endpoint
# =========================================
@router.post(
        "/read",
        response_model=ReadOut,
        status_code=status.HTTP_200_OK,
        summary="OCR + NER (GLiNER) with document class"
)
async def perform_read(
    request: Request,
    read_params: ReadIn = Depends(),
    file: UploadFile = File(
        None,
        description="Upload image or PDF file (optional)"
    ),
) -> ReadOut:

    start = time.perf_counter()

    try:
        content, filenames = await get_documents(request, file)
        predictor = init_predictor(read_params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    results = predictor(content, filenames)
    result: ReadOut = results[0]

    return result.model_copy(
        update={
            "name": filenames[0],
            "duration": round(time.perf_counter() - start, 4)
        }
    )