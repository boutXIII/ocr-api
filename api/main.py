# =========================================
# Imports
# =========================================
import time

from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi

from api import config as cfg
from api.routes import ocr, health

# =========================================
# FastAPI App
# =========================================
app = FastAPI(
    title=cfg.PROJECT_NAME,
    description=cfg.PROJECT_DESCRIPTION,
    debug=cfg.DEBUG,
    version=cfg.VERSION
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
    response.headers["X-Process-Time"] = str(process_time)
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