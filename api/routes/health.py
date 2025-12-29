# =========================================
# Imports
# =========================================
import time
from fastapi import APIRouter, status

from api.schemas import HealthOut
from api.state import app_state
from api.utils.tools import TimedResponseBuilder

# =========================================
# Router
# =========================================
router = APIRouter()

# =========================================
# Endpoint
# =========================================
@router.get(
    "/",
    response_model=HealthOut,
    status_code=status.HTTP_200_OK,
    summary="Return API and model status",
)
async def health_check() :
    start = time.perf_counter()

    engine = app_state["engineType"]
    models_available = app_state["models_available"]
    device = app_state["device"]

    duration = time.perf_counter() - start
    return TimedResponseBuilder.build(
        HealthOut,
        duration,
        status="ok",
        engine=engine,
        models_available=models_available,
        device=device,
    )