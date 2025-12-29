# =========================================
# Imports
# =========================================
from api.config import *
import os
import time

from fastapi.concurrency import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi

from api.logger import get_logger
from api.bootstrap import preload_models
from api import config as cfg
from api.routes import ocr, health

# =========================================
# Logger (UNE seule fois)
# =========================================
logger = get_logger("BOOT")

# =========================================
# Lifespan (startup / shutdown)
# =========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🔥 STARTUP
    logger.info("=== LOGGER STARTED ===")
    logger.info(f"ONNXTR_CACHE_DIR set to {os.environ.get('ONNXTR_CACHE_DIR')}")

    logger.info("=== CACHE AND MODELS PRELOADING ===")
    preload_models()

    yield  # 👉 application runs here

    # 🔻 SHUTDOWN (optionnel)
    logger.info("=== APPLICATION SHUTDOWN ===")

# =========================================
# FastAPI App
# =========================================
app = FastAPI(
    title=cfg.PROJECT_NAME,
    description=cfg.PROJECT_DESCRIPTION,
    debug=cfg.DEBUG,
    version=cfg.VERSION,
    lifespan=lifespan,
)

# =========================================
# Routing
# =========================================
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(ocr.router, prefix="/ocr", tags=["ocr"])

# =========================================
# Middleware
# =========================================
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Execution-Time"] = str(process_time)
    return response

# =========================================
# OpenAPI override
# =========================================
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=cfg.PROJECT_NAME,
        version=cfg.VERSION,
        description=cfg.PROJECT_DESCRIPTION,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi