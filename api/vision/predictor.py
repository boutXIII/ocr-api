# =========================================
# Imports
# =========================================
from collections.abc import Callable

import cv2
import numpy as np
import onnxruntime as ort

from onnxtr.models.builder import DocumentBuilder
from onnxtr.models import ocr_predictor, detection_predictor, page_orientation_predictor
from onnxtr.models._utils import estimate_orientation
from onnxtr.models._utils import get_language

from api.utils.tools import load_det_models, load_reco_models, load_orientation_models, resolve_geometry
from api.schemas import OCRBlock, OCRIn, OCRLine, OCROut, OCRPage, OCRWord
from api.vision.print_type import classify_print_type

from api.logger import get_logger
logger = get_logger("PREDICTOR")

# =========================================
# CONFIG
# =========================================
PRINT_TYPE_ONNX = "printed_handwritten_classifier/printed_handwritten_resnet18.onnx"
IMG_SIZE = (64, 256)
CLASSES = ["handwritten", "printed"]

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

def preprocess_for_onnx(crop_bgr):
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    return img[None, ...]

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

    # ---------------------------------
    # 1. Récupération brute des paramètres
    # ---------------------------------
    params = request.model_dump()

    logger.debug(f"Predictor params: {params}")

    use_print_type = params.pop("use_print_type", False)
    reco_printed = params.pop("reco_printed", None)
    reco_handwritten = params.pop("reco_handwritten", None)

    bin_thresh = params.pop("bin_thresh")
    box_thresh = params.pop("box_thresh")

    logger.debug(f"Predictor params: {params}")

    if isinstance(request, (OCRIn)):
        if use_print_type:
            logger.info("Print-type based OCR mode")

            all_boxes = []
            all_scores = []
            all_text_preds = []
            all_crop_orientations = []
            page_shapes = []
            pages = []
            all_print_types = []

            detector = detection_predictor(
                arch=load_det_models(params["det_arch"]),
                assume_straight_pages=False,
            )

            # page_orient_predictor = page_orientation_predictor("mobilenet_v3_small_page_orientation")

            page_orient_predictor = load_orientation_models()

            reco_printed_model = load_reco_models(reco_printed)
            reco_handwritten_model = load_reco_models(reco_handwritten)

            def predictor(content, filenames):
                det_out, out_maps = detector(content, return_maps=True)

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

                    texts = [t for t, conf in all_text_preds[pi] if t]
                    full_text = " ".join(texts)
                    lang, lang_conf = get_language(full_text)
                    logger.debug(f"Detected language: {lang} with confidence {lang_conf}")
                    
                    results.append(
                        OCROut(
                            name=filename,
                            orientation={
                                "value": orientations[0].get("value"),
                                "confidence": orientations[0].get("confidence"),
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
            params["detect_language"] = True
            params["assume_straight_pages"] = False
            params["straighten_pages"] = True
            params["detect_orientation"] = True
            predictor_core = ocr_predictor(**params)
            predictor_core.det_predictor.model.postprocessor.bin_thresh = bin_thresh
            predictor_core.det_predictor.model.postprocessor.box_thresh = box_thresh
            

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