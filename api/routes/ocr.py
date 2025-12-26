# =========================================
# Imports
# =========================================
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
        response_model=list[OCROut],
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
) -> list[OCROut]:
    """Runs docTR OCR model to analyze the input image"""
    try:
        # generator object to list
        content, filenames = await get_documents(request, file)
        predictor = init_predictor(ocr_params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return predictor(content, filenames)

# =========================================
# OCR + NER endpoint
# =========================================
@router.post(
        "/read",
        response_model=list[ReadOut],
        status_code=status.HTTP_200_OK,
        summary="OCR + NER (GLiNER) with document class"
)
async def perform_ocr(
    request: Request,
    read_params: ReadIn = Depends(),
    file: UploadFile = File(
        None,
        description="Upload image or PDF file (optional)"
    ),
) -> list[ReadOut]:

    try:
        content, filenames = await get_documents(request, file)
        predictor = init_predictor(read_params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return predictor(content, filenames)