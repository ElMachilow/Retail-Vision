import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging
from app.db.connection import init_db
from app.schemas.product import ErrorResponse

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
STATIC_DIR = WEB_DIR / "static"


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db(settings)
    cors_origins = [
        origin.strip()
        for origin in settings.cors_allow_origins.split(",")
        if origin.strip()
    ] or ["*"]

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "MVP para registrar productos retail peruanos desde imágenes, "
            "usando YOLO para detectar la región relevante y PaddleOCR para extraer texto."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def product_screen() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/productos", include_in_schema=False)
    async def products_screen() -> FileResponse:
        return FileResponse(WEB_DIR / "productos.html")

    @app.get("/admin", include_in_schema=False)
    async def admin_screen() -> FileResponse:
        return FileResponse(WEB_DIR / "admin.html")

    register_exception_handlers(app)
    return app


def register_exception_handlers(app: FastAPI) -> None:
    logger = logging.getLogger(__name__)

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        trace_id = request.headers.get("X-Trace-ID")
        logger.warning(
            "error esperado code=%s message=%s",
            exc.error_code,
            exc.message,
            extra={"trace_id": trace_id or "-"},
        )
        payload = ErrorResponse(
            trace_id=trace_id,
            error_code=exc.error_code,
            message=exc.message,
            detail=exc.detail,
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        trace_id = request.headers.get("X-Trace-ID")
        logger.exception(
            "error no controlado",
            extra={"trace_id": trace_id or "-"},
        )
        payload = ErrorResponse(
            trace_id=trace_id,
            error_code="UNHANDLED_ERROR",
            message="Ocurrió un error no controlado durante el procesamiento.",
            detail=str(exc) if get_settings().expose_internal_errors else None,
        )
        return JSONResponse(status_code=500, content=payload.model_dump())


app = create_app()
