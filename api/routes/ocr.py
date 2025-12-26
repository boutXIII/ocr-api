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
    OCRIn, OCROut, OCRPage, OCRBlock, OCRLine, OCRWord,
    ReadIn, ReadOut, EntityOut, FieldResult,
)
from api.utils.tools import get_documents, resolve_geometry
from api.vision.predictor import init_predictor
from api.ner.strategy import build_registry
from api.ner.extractor import extract_entities

# =========================================
# Router & Registry
# =========================================
router = APIRouter()
REGISTRY = build_registry()

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
    ocr_params: OCRIn = Depends(),
    read_params: ReadIn = Depends(),
    file: UploadFile = File(
        None,
        description="Upload image or PDF file (optional)"
    ),
) -> list[ReadOut]:

    try:
        content, filenames = await get_documents(request, file)
        predictor = init_predictor(ocr_params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    out = predictor(content, filenames)

    logger.debug(f"OCR output: {out}")

    # out = predictor(content).export()

    results: list[ReadOut] = []   # ✅ AVANT LA BOUCLE

    for page, filename in zip(out.get("pages", []), filenames):

        # --- texte OCR ---
        text = " ".join(
            word["value"]
            for block in page.get("blocks", [])
            for line in block.get("lines", [])
            for word in line.get("words", [])
        )

        try:
            entities_raw = extract_entities(
                text=text,
                document_class=read_params.document_class,
                gliner_threshold=read_params.gliner_threshold,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        ctx: dict = {}
        validated_fields: list[FieldResult] = []

        for e in entities_raw:
            label = e["label"]
            raw = e["text"]
            score = float(e["score"])

            if label == "person":
                ctx["person"] = raw

            evaluated = REGISTRY.evaluate(
                document_class=read_params.document_class,
                label=label,
                raw_value=raw,
                score=score,
                ctx=ctx,
            )

            validated_fields.append(FieldResult(**evaluated))

        results.append(
            ReadOut(
                name=filename,
                text=text,
                entities=[
                    EntityOut(
                        label=e["label"],
                        text=e["text"],
                        start=e["start"],
                        end=e["end"],
                        score=e["score"],
                    )
                    for e in entities_raw
                ],
                fields_validated=validated_fields,
            )
        )

    return results