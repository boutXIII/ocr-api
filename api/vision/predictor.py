# =========================================
# Imports
# =========================================
from collections.abc import Callable

from onnxtr.models import ocr_predictor

from api.utils.documents import load_reco_models
from api.schemas import OCRIn

from api.logger import get_logger
logger = get_logger("PREDICTOR")

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
    params = request.model_dump()
    params["det_arch"] = params.get("det_arch", "db_resnet50")
    params["reco_arch"] = params.get("reco_arch", "crnn_vgg16_bn")

    use_print_type = params.pop("use_print_type", True)
    reco_printed = params.pop("reco_printed", "crnn_vgg16_bn")
    reco_handwritten = params.pop("reco_handwritten", "parseq")

    bin_thresh = params.pop("bin_thresh", 0.3)
    box_thresh = params.pop("box_thresh", 0.1)

    predictor = ocr_predictor(**params)
    predictor.det_predictor.model.postprocessor.bin_thresh = bin_thresh
    predictor.det_predictor.model.postprocessor.box_thresh = box_thresh
    
    if isinstance(request, (OCRIn)):
        if use_print_type:
                predictor.use_print_type = True
                predictor.reco_printed_model, predictor.reco_handwritten_model = load_reco_models(
                    reco_printed, reco_handwritten
            )
        return predictor