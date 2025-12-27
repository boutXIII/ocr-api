# =========================================
# Imports
# =========================================
from collections.abc import Callable

import cv2
from fastapi import HTTPException
import numpy as np

from onnxtr.models.builder import DocumentBuilder
from onnxtr.models import detection_predictor
from onnxtr.models._utils import estimate_orientation
from onnxtr.models._utils import get_language

from api.ner.extractor import extract_entities
from api.ner.strategy import build_registry
from api.utils.tools import load_det_models, load_predictor, load_reco_models, load_page_orientation_models, resolve_geometry
from api.schemas import EntityOut, FieldResult, OCRBlock, OCRIn, OCRLine, OCROut, OCRPage, OCRWord, PredictMode, ReadIn, ReadOut
from api.vision.print_type import classify_print_type
from api.vision.model_store import ModelStore

from api.logger import get_logger
logger = get_logger("PREDICTOR")


REGISTRY = build_registry()
# =========================================
# Utils
# =========================================
def to_absolute_bbox(coords, img_shape):
    h, w = img_shape

    # Straight box: [x1, y1, x2, y2]
    if len(coords) == 4:
        x1, y1, x2, y2 = coords
        return (
            int(x1 * w),
            int(y1 * h),
            int(x2 * w),
            int(y2 * h),
        )

    # Rotated box: [x1,y1,x2,y2,x3,y3,x4,y4]
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    xs = pts[:, 0] * w
    ys = pts[:, 1] * h

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    )

def recognize(crop_bgr, rec_model):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.resize(crop_rgb, (128, 32))
    crop_rgb = crop_rgb.astype(np.float32) / 255.0
    crop_rgb = (crop_rgb - 0.5) / 0.5
    crop_rgb = np.transpose(crop_rgb, (2, 0, 1))
    crop_rgb = crop_rgb[None, ...]
    result = rec_model(crop_rgb)

    logger.debug(f"Recognition result: {result}")

    if isinstance(result, dict) and "preds" in result and len(result["preds"]) > 0:
        text, conf = result["preds"][0]
        return text, float(conf)
    
    return "", 0.0

def iou(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)

    return inter / (area_a + area_b - inter)

def find_meta(word_geom, registry):
    best = None
    best_iou = 0.0

    for r in registry:
        score = iou(word_geom, r["bbox"])
        if score > best_iou:
            best_iou = score
            best = r

    return best

def build_word(word, all_print_types, pi):
    geom = resolve_geometry(word["geometry"])
    meta = find_meta(geom, all_print_types[pi]) or {}

    return OCRWord(
        value=word["value"],
        geometry=geom,
        objectness_score=1.0,
        confidence=round(word.get("confidence", 1.0), 2),
        crop_orientation={"value": 0, "confidence": None},
        print_type=meta.get("print_type", "unknown"),
        print_confidence=meta.get("print_confidence", 0.0),
    )

def call_print_type_model(params, reco_printed, reco_handwritten):

    page_orient_predictor = None
    if params.get("detect_orientation", False):
        logger.debug("Loading page orientation model")
        page_orient_predictor = load_page_orientation_models()
        detector = detection_predictor(
            arch=load_det_models(params["det_arch"]),
            assume_straight_pages=False,
        )
    else:
        detector = detection_predictor(
            arch=load_det_models(params["det_arch"]),
            assume_straight_pages=True,
        )

    reco_printed_model = load_reco_models(reco_printed)
    reco_handwritten_model = load_reco_models(reco_handwritten)

    return detector, reco_handwritten_model, reco_printed_model, page_orient_predictor

def build_readOut(out, filenames, document_class, gliner_threshold) -> list[ReadOut]:
    results: list[ReadOut] = []
    for page, filename in zip(out.get("pages", []), filenames):
            # --- texte OCR ---
        text = " ".join(
            word["value"]
            for block in page.get("blocks", [])
            for line in block.get("lines", [])
            for word in line.get("words", [])
        )
        logger.debug(f"Extracted text for NER: {text}")

        try:
            entities_raw = extract_entities(
                text=text,
                document_class=document_class,
                gliner_threshold=gliner_threshold,
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
                document_class=document_class,
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

# =========================================
# Predictor factory
# =========================================
def init_predictor(request: OCRIn ) -> Callable:
    """Initialize the predictor based on the request

    Args:
        request: input request

    Returns:
        Callable: the predictor
    """
    logger.debug("init_predictor")

    if request.mode == PredictMode.OCR:
        logger.info("OCR mode selected.")
        return _init_ocr_predictor(request)
    elif request.mode == PredictMode.READ:
        logger.info("Read/NER mode selected.")
        return _init_read_predictor(request)
    
def _init_ocr_predictor(request: OCRIn) -> Callable:
    """Initialize the OCR predictor based on the request

    Args:
        request: input request

    Returns:
        Callable: the OCR predictor
    """
    logger.debug("Initializing OCR predictor")

    params = request.model_dump()
    use_print_type = params.pop("use_print_type", False)
    reco_printed = params.pop("reco_printed", None)
    reco_handwritten = params.pop("reco_handwritten", None)

    logger.debug(f"Predictor params: {params}")

    if use_print_type:
        logger.info("Print-type based OCR mode")

        all_boxes = []
        all_scores = []
        all_text_preds = []
        all_crop_orientations = []
        page_shapes = []
        pages = []
        all_print_types = []

        detector, reco_handwritten_model, reco_printed_model, page_orient_predictor = call_print_type_model(params, reco_printed, reco_handwritten)

        def predictor(content, filenames):
            det_out, out_maps = detector(content, return_maps=True)
            orientations = []

            if params.get("detect_orientation", False):
                logger.debug("Estimating page orientations")
                seg_maps = [
                    np.where(out_map > getattr(detector.model.postprocessor, "bin_thresh"), 255, 0).astype(np.uint8)
                    for out_map in out_maps
                ]

                _, classes, probs = zip(page_orient_predictor(content))
                # Flatten to list of tuples with (value, confidence)
                page_orientations = [
                    (orientation, prob)
                    for page_classes, page_probs in zip(classes, probs)
                    for orientation, prob in zip(page_classes, page_probs)
                ]

                origin_pages_orientations = [
                    estimate_orientation(seq_map, general_orientation)
                    for seq_map, general_orientation in zip(seg_maps, page_orientations)
                ]
                orientations = [
                    {"value": orientation, "confidence": prob} for orientation, prob in zip(origin_pages_orientations, probs[0])
                ]
                logger.debug(f"orientations: {orientations}")

            for doc, res, filename in zip(content, det_out, filenames):
                img_shape = doc.shape[:2]
                page_shapes.append(img_shape)
                pages.append(doc)

                boxes = []
                scores = []
                text_preds = []
                crop_orients = []
                print_types = []

                for det in res:
                    det = np.asarray(det)
                    coords = det[:-1]

                    # Géométrie normalisée
                    if len(coords) == 4:
                        x1, y1, x2, y2 = coords
                        geom = [[x1, y1], [x2, y2]]
                    else:
                        geom = np.array(coords).reshape(-1, 2).tolist()

                    px1, py1, px2, py2 = to_absolute_bbox(geom, img_shape)

                    crop = doc[py1:py2, px1:px2]
                    if crop.size == 0:
                        continue

                    ptype, pscore = classify_print_type(crop)

                    reco_model = (
                        reco_printed_model
                        if ptype == "printed"
                        else reco_handwritten_model
                    )

                    text, conf = recognize(crop, reco_model)

                    H, W = img_shape
                    boxes.append([
                        px1 / W,
                        py1 / H,
                        px2 / W,
                        py2 / H,
                    ])
                    scores.append(pscore)
                    text_preds.append((text, conf))
                    crop_orients.append({
                        "value": 0,
                        "confidence": None,
                    })
                    print_types.append({
                        "bbox": [px1 / W, py1 / H, px2 / W, py2 / H],
                        "print_type": ptype,
                        "print_confidence": round(pscore, 4)
                    })
                    if ptype == "handwritten":
                        logger.debug(f"Crop recognized as {ptype} with confidence {pscore}, text: {text} (conf: {conf})")

                all_boxes.append(np.array(boxes, dtype=np.float32))
                all_scores.append(np.array(scores, dtype=np.float32))
                all_text_preds.append(text_preds)
                all_crop_orientations.append(crop_orients)
                all_print_types.append(print_types)


                builder = DocumentBuilder(
                    resolve_lines=True,
                    resolve_blocks=False,
                    paragraph_break=0.0035,
                )

                predictor_core = builder(
                    pages=pages,
                    boxes=all_boxes,
                    objectness_scores=all_scores,
                    text_preds=all_text_preds,
                    page_shapes=page_shapes,
                    crop_orientations=all_crop_orientations,
                )

            out = predictor_core.export()
            results: list[OCROut] = []

            for pi, (page, filename) in enumerate(zip(out.get("pages", []), filenames)):

                if params.get("detect_language", False):
                    logger.debug("Estimating page language")
                    texts = [t for t, conf in all_text_preds[pi] if t]
                    full_text = " ".join(texts)
                    lang, lang_conf = get_language(full_text)
                    logger.debug(f"Detected language: {lang} with confidence {lang_conf}")
                else:
                    lang, lang_conf = "unknown", 0.0
                
                results.append(
                    OCROut(
                        name=filename,
                        orientation={
                            "value": orientations[0].get("value") if orientations else 0,
                            "confidence": orientations[0].get("confidence") if orientations else None,
                        },
                        language={
                            "value": lang,
                            "confidence": lang_conf,
                        },
                        dimensions=tuple(page.get("dimensions", (0, 0))),
                        items=[
                            OCRPage(
                                blocks=[
                                    OCRBlock(
                                        geometry=resolve_geometry(block["geometry"]),
                                        objectness_score=1.0,
                                        lines=[
                                            OCRLine(
                                                geometry=resolve_geometry(line["geometry"]),
                                                objectness_score=1.0,
                                                words=[
                                                    build_word(word, all_print_types, pi)
                                                    for word in line.get("words", [])
                                                ],
                                            )
                                            for line in block.get("lines", [])
                                        ],
                                    )
                                    for block in page.get("blocks", [])
                                ]
                            )
                        ],
                    )
                )

            return results

    else:
        logger.info("Simple OCR mode")

        predictor_core = load_predictor(**params)

        def predictor(content, filenames):
            out = predictor_core(content).export()
            logger.debug(f"Detected orientation: {out['pages'][0]['orientation']['value']} degrees")
            logger.debug(f"Detected orientation: {out['pages'][0]['orientation']['confidence']} degrees")

            results: list[OCROut] = [
                OCROut(
                    name=filename,
                    orientation={
                        "value": page.get("orientation", {}).get("value"),
                        "confidence": page.get("orientation", {}).get("confidence"),
                    },
                    language={
                        "value": page.get("language", {}).get("value"),
                        "confidence": page.get("language", {}).get("confidence")
                    },
                    dimensions=tuple(page.get("dimensions", (0, 0))),
                    items=[
                        OCRPage(
                            blocks=[
                                OCRBlock(
                                    geometry=resolve_geometry(block["geometry"]),
                                    objectness_score=1.0,
                                    lines=[
                                        OCRLine(
                                            geometry=resolve_geometry(line["geometry"]),
                                            objectness_score=1.0,
                                            words=[
                                                OCRWord(
                                                    value=word["value"],
                                                    geometry=resolve_geometry(word["geometry"]),
                                                    objectness_score=1.0,
                                                    confidence=round(
                                                        word.get("confidence", 1.0), 2
                                                    ),
                                                    crop_orientation={
                                                        "value": 0,
                                                        "confidence": None,
                                                    },
                                                    print_type=None,
                                                    print_confidence=None,
                                                )
                                                for word in line.get("words", [])
                                            ],
                                        )
                                        for line in block.get("lines", [])
                                    ],
                                )
                                for block in page.get("blocks", [])
                            ]
                        )
                    ],
                )
                for page, filename in zip(out.get("pages", []), filenames)
            ]

            return results
    
    return predictor

def _init_read_predictor(request: ReadIn) -> Callable:
    """Initialize the Read/NER predictor based on the request

    Args:
        request: input request

    Returns:
        Callable: the Read/NER predictor
    """
    logger.debug("Initializing Read/NER predictor")

    params = request.model_dump()
    use_print_type = params.pop("use_print_type", False)
    reco_printed = params.pop("reco_printed", None)
    reco_handwritten = params.pop("reco_handwritten", None)

    document_class = params.pop("document_class", "FACT_MEDECINE_DOUCE")
    gliner_threshold = params.pop("gliner_threshold", 0.7)

    logger.debug(f"Predictor params: {params}")

    if use_print_type:
        logger.info("Print-type based Read/NER mode")

        all_boxes = []
        all_scores = []
        all_text_preds = []
        all_crop_orientations = []
        page_shapes = []
        pages = []
        all_print_types = []

        detector, reco_handwritten_model, reco_printed_model, page_orient_predictor = call_print_type_model(params, reco_printed, reco_handwritten)

        def predictor(content, filenames):
            det_out, out_maps = detector(content, return_maps=True)
            orientations = []

            if params.get("detect_orientation", False):
                logger.debug("Estimating page orientations")
                seg_maps = [
                    np.where(out_map > getattr(detector.model.postprocessor, "bin_thresh"), 255, 0).astype(np.uint8)
                    for out_map in out_maps
                ]

                _, classes, probs = zip(page_orient_predictor(content))
                # Flatten to list of tuples with (value, confidence)
                page_orientations = [
                    (orientation, prob)
                    for page_classes, page_probs in zip(classes, probs)
                    for orientation, prob in zip(page_classes, page_probs)
                ]

                origin_pages_orientations = [
                    estimate_orientation(seq_map, general_orientation)
                    for seq_map, general_orientation in zip(seg_maps, page_orientations)
                ]
                orientations = [
                    {"value": orientation, "confidence": prob} for orientation, prob in zip(origin_pages_orientations, probs[0])
                ]
                logger.debug(f"orientations: {orientations}")

            for doc, res in zip(content, det_out):
                img_shape = doc.shape[:2]
                page_shapes.append(img_shape)
                pages.append(doc)

                boxes = []
                scores = []
                text_preds = []
                crop_orients = []
                print_types = []

                for det in res:
                    det = np.asarray(det)
                    coords = det[:-1]

                    # Géométrie normalisée
                    if len(coords) == 4:
                        x1, y1, x2, y2 = coords
                        geom = [[x1, y1], [x2, y2]]
                    else:
                        geom = np.array(coords).reshape(-1, 2).tolist()

                    px1, py1, px2, py2 = to_absolute_bbox(geom, img_shape)

                    crop = doc[py1:py2, px1:px2]
                    if crop.size == 0:
                        continue

                    ptype, pscore = classify_print_type(crop)

                    reco_model = (
                        reco_printed_model
                        if ptype == "printed"
                        else reco_handwritten_model
                    )

                    text, conf = recognize(crop, reco_model)

                    H, W = img_shape
                    boxes.append([
                        px1 / W,
                        py1 / H,
                        px2 / W,
                        py2 / H,
                    ])
                    scores.append(pscore)
                    text_preds.append((text, conf))
                    crop_orients.append({
                        "value": 0,
                        "confidence": None,
                    })
                    print_types.append({
                        "bbox": [px1 / W, py1 / H, px2 / W, py2 / H],
                        "print_type": ptype,
                        "print_confidence": round(pscore, 4)
                    })
                    if ptype == "handwritten":
                        logger.debug(f"Crop recognized as {ptype} with confidence {pscore}, text: {text} (conf: {conf})")

                all_boxes.append(np.array(boxes, dtype=np.float32))
                all_scores.append(np.array(scores, dtype=np.float32))
                all_text_preds.append(text_preds)
                all_crop_orientations.append(crop_orients)
                all_print_types.append(print_types)


                builder = DocumentBuilder(
                    resolve_lines=True,
                    resolve_blocks=False,
                    paragraph_break=0.0035,
                )

                predictor_core = builder(
                    pages=pages,
                    boxes=all_boxes,
                    objectness_scores=all_scores,
                    text_preds=all_text_preds,
                    page_shapes=page_shapes,
                    crop_orientations=all_crop_orientations,
                )

            out = predictor_core.export()
            logger.debug(f"Read/NER output: {out}")

            return build_readOut(out, filenames, document_class, gliner_threshold)
    else:
        logger.info("Simple Read/NER mode")

        predictor_core = load_predictor(**params)

        def predictor(content, filenames):
            out = predictor_core(content).export()
            logger.debug(f"Read/NER output: {out}")

            return build_readOut(out, filenames, document_class, gliner_threshold)

    return predictor