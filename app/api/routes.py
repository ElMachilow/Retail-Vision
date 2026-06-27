from uuid import uuid4

from fastapi import APIRouter, Depends, File, Header, Query, Response, UploadFile

from app.api.dependencies import (
    get_inventory_repository,
    get_product_pipeline,
    get_product_repository,
    get_product_categorizer,
    get_recognition_repository,
    get_suggestion_service,
)
from app.core.config import Settings, get_settings
from app.core.exceptions import InvalidImageError
from app.core.security import require_admin
from app.repositories.inventory import (
    InventoryItemRecord,
    InventoryRepository,
    InventorySessionRecord,
    ProductStockCountRecord,
)
from app.repositories.products import ProductRecord, ProductRepository
from app.repositories.recognitions import RecognitionEventRecord, RecognitionRepository
from app.schemas.product import (
    ErrorResponse,
    InventoryItemCreateRequest,
    InventoryItemResponse,
    InventoryItemsResponse,
    InventoryRecognizeResponse,
    InventorySessionCreateRequest,
    InventorySessionResponse,
    InventorySessionsResponse,
    InventorySummaryResponse,
    ProductCategorizeRequest,
    ProductCategorizeResponse,
    ProductCreateRequest,
    ProductListResponse,
    ProductRecognitionResponse,
    ProductResponse,
    ProductSuggestionItemSchema,
    ProductSuggestionsResponse,
    ProductStockCountCreateRequest,
    ProductStockCountResponse,
    ProductUpdateRequest,
    RecognitionEventResponse,
    RecognitionEventsResponse,
    RecognitionReviewRequest,
    RecognitionStatsResponse,
)
from app.services.categorizer import ProductCategorizer
from app.services.pipeline import ProductRecognitionPipeline
from app.services.suggestions import ProductSuggestionService

router = APIRouter()


def _to_response(record: ProductRecord) -> ProductResponse:
    return ProductResponse(
        id=record.id,
        nombre_producto=record.nombre_producto,
        marca=record.marca,
        tipo_producto=record.tipo_producto,
        presentacion=record.presentacion,
        contenido_neto=record.contenido_neto,
        unidad_medida=record.unidad_medida,
        categoria_sugerida=record.categoria_sugerida,
        codigo_barras=record.codigo_barras,
        precio_venta=record.precio_venta,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _inventory_session_to_response(record: InventorySessionRecord) -> InventorySessionResponse:
    return InventorySessionResponse(
        id=record.id,
        nombre=record.nombre,
        estado=record.estado,
        created_at=record.created_at,
        closed_at=record.closed_at,
    )


def _inventory_item_to_response(record: InventoryItemRecord) -> InventoryItemResponse:
    return InventoryItemResponse(
        id=record.id,
        session_id=record.session_id,
        product_id=record.product_id,
        recognition_event_id=record.recognition_event_id,
        nombre_producto=record.nombre_producto,
        marca=record.marca,
        tipo_producto=record.tipo_producto,
        categoria=record.categoria,
        contenido_neto=record.contenido_neto,
        unidad_medida=record.unidad_medida,
        cantidad=record.cantidad,
        ubicacion=record.ubicacion,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _product_stock_count_to_response(record: ProductStockCountRecord) -> ProductStockCountResponse:
    return ProductStockCountResponse(
        id=record.id,
        mobile_product_id=record.mobile_product_id,
        nombre_producto=record.nombre_producto,
        cantidad_final=record.cantidad_final,
        confianza=record.confianza,
        total_fotos=record.total_fotos,
        valid_fotos=record.valid_fotos,
        source=record.source,
        created_at=record.created_at,
        photos=[
            {
                "id": photo.id,
                "recognition_event_id": photo.recognition_event_id,
                "source_name": photo.source_name,
                "detected_name": photo.detected_name,
                "matched": photo.matched,
                "accepted": photo.accepted,
                "confidence": photo.confidence,
                "warnings": photo.warnings,
                "created_at": photo.created_at,
            }
            for photo in record.photos
        ],
    )


def _recognition_to_response(
    record: RecognitionEventRecord,
    categorizer: ProductCategorizer | None = None,
) -> RecognitionEventResponse:
    predicted_marca = record.predicted_marca
    predicted_tipo_producto = record.predicted_tipo_producto
    predicted_presentacion = record.predicted_presentacion
    predicted_contenido_neto = record.predicted_contenido_neto
    predicted_unidad_medida = record.predicted_unidad_medida
    predicted_categoria_sugerida = record.predicted_categoria_sugerida
    if categorizer and not (record.final_categoria_sugerida or predicted_categoria_sugerida):
        name = record.final_nombre_producto or record.predicted_nombre_producto
        if name:
            categorized = categorizer.categorize(name, context_text=record.ocr_text)
            predicted_marca = predicted_marca or categorized.marca
            predicted_tipo_producto = predicted_tipo_producto or categorized.tipo_producto
            predicted_presentacion = predicted_presentacion or categorized.presentacion
            predicted_contenido_neto = predicted_contenido_neto or categorized.contenido_neto
            predicted_unidad_medida = predicted_unidad_medida or categorized.unidad_medida
            predicted_categoria_sugerida = categorized.categoria_sugerida

    return RecognitionEventResponse(
        id=record.id,
        trace_id=record.trace_id,
        source_name=record.source_name,
        image_url=f"/api/v1/admin/reconocimientos/{record.id}/image",
        status=record.status,
        predicted_nombre_producto=record.predicted_nombre_producto,
        predicted_marca=predicted_marca,
        predicted_tipo_producto=predicted_tipo_producto,
        predicted_presentacion=predicted_presentacion,
        predicted_contenido_neto=predicted_contenido_neto,
        predicted_unidad_medida=predicted_unidad_medida,
        predicted_categoria_sugerida=predicted_categoria_sugerida,
        final_nombre_producto=record.final_nombre_producto,
        final_marca=record.final_marca,
        final_tipo_producto=record.final_tipo_producto,
        final_presentacion=record.final_presentacion,
        final_contenido_neto=record.final_contenido_neto,
        final_unidad_medida=record.final_unidad_medida,
        final_categoria_sugerida=record.final_categoria_sugerida,
        final_codigo_barras=record.final_codigo_barras,
        yolo_confidence=record.yolo_confidence,
        yolo_class_name=record.yolo_class_name,
        ocr_confidence=record.ocr_confidence,
        ocr_text=record.ocr_text,
        warnings=record.warnings,
        bbox=record.bbox,
        failure_reason=record.failure_reason,
        review_notes=record.review_notes,
        use_for_training=record.use_for_training,
        linked_product_id=record.linked_product_id,
        recognition=record.recognition,
        reviewed_at=record.reviewed_at,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.get("/health", tags=["health"])
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.app_env,
    }


@router.post(
    "/inventory/sessions",
    response_model=InventorySessionResponse,
    status_code=201,
    tags=["inventory"],
    summary="Crea una sesion de inventario por foto.",
)
def create_inventory_session(
    payload: InventorySessionCreateRequest,
    repository: InventoryRepository = Depends(get_inventory_repository),
) -> InventorySessionResponse:
    return _inventory_session_to_response(repository.create_session(payload.nombre))


@router.get(
    "/inventory/sessions",
    response_model=InventorySessionsResponse,
    tags=["inventory"],
    summary="Lista sesiones de inventario.",
)
def list_inventory_sessions(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repository: InventoryRepository = Depends(get_inventory_repository),
) -> InventorySessionsResponse:
    return InventorySessionsResponse(
        items=[_inventory_session_to_response(item) for item in repository.list_sessions(limit, offset)]
    )


@router.put(
    "/inventory/sessions/{session_id}/close",
    response_model=InventorySessionResponse,
    tags=["inventory"],
    summary="Cierra una sesion de inventario.",
)
def close_inventory_session(
    session_id: int,
    repository: InventoryRepository = Depends(get_inventory_repository),
) -> InventorySessionResponse:
    return _inventory_session_to_response(repository.close_session(session_id))


@router.post(
    "/inventory/sessions/{session_id}/items/recognize",
    response_model=InventoryRecognizeResponse,
    tags=["inventory"],
    summary="Reconoce un producto desde foto para agregarlo al inventario.",
)
async def recognize_inventory_item(
    session_id: int,
    image: UploadFile = File(..., description="Imagen JPG, PNG o WEBP del producto a contar."),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
    settings: Settings = Depends(get_settings),
    inventory: InventoryRepository = Depends(get_inventory_repository),
    products: ProductRepository = Depends(get_product_repository),
    pipeline: ProductRecognitionPipeline = Depends(get_product_pipeline),
    recognitions: RecognitionRepository = Depends(get_recognition_repository),
) -> InventoryRecognizeResponse:
    inventory.get_session(session_id)
    trace_id = x_trace_id or str(uuid4())
    content_type = (image.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise InvalidImageError("El archivo enviado debe ser una imagen.")

    image_bytes = await image.read()
    max_bytes = settings.max_image_mb * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise InvalidImageError(f"La imagen supera el limite de {settings.max_image_mb} MB.")

    result = pipeline.process(image_bytes=image_bytes, trace_id=trace_id, source_name=image.filename)
    event = recognitions.create_from_response(
        image_bytes=image_bytes,
        content_type=image.content_type,
        source_name=image.filename,
        response=result,
    )
    matches = products.search_by_name(result.producto.nombre_producto or "", limit=1) if result.producto.nombre_producto else []
    return InventoryRecognizeResponse(
        trace_id=result.trace_id,
        producto=result.producto,
        recognition_event_id=event.id,
        image_url=f"/api/v1/admin/reconocimientos/{event.id}/image",
        matching_product_id=matches[0].id if matches else None,
        warnings=result.warnings,
        processing_ms=result.processing_ms,
    )


@router.post(
    "/inventory/sessions/{session_id}/items",
    response_model=InventoryItemResponse,
    status_code=201,
    tags=["inventory"],
    summary="Guarda una linea de conteo en una sesion de inventario.",
)
def create_inventory_item(
    session_id: int,
    payload: InventoryItemCreateRequest,
    repository: InventoryRepository = Depends(get_inventory_repository),
    categorizer: ProductCategorizer = Depends(get_product_categorizer),
) -> InventoryItemResponse:
    data = payload.model_dump()
    if not data.get("categoria"):
        categorized = categorizer.categorize(data["nombre_producto"])
        data["categoria"] = categorized.categoria_sugerida
        data["marca"] = data.get("marca") or categorized.marca
        data["tipo_producto"] = data.get("tipo_producto") or categorized.tipo_producto
        data["contenido_neto"] = data.get("contenido_neto") or categorized.contenido_neto
        data["unidad_medida"] = data.get("unidad_medida") or categorized.unidad_medida
    return _inventory_item_to_response(repository.create_item(session_id, data))


@router.get(
    "/inventory/sessions/{session_id}/items",
    response_model=InventoryItemsResponse,
    tags=["inventory"],
    summary="Lista productos contados en una sesion de inventario.",
)
def list_inventory_items(
    session_id: int,
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repository: InventoryRepository = Depends(get_inventory_repository),
) -> InventoryItemsResponse:
    return InventoryItemsResponse(
        items=[_inventory_item_to_response(item) for item in repository.list_items(session_id, limit, offset)]
    )


@router.get(
    "/inventory/sessions/{session_id}/summary",
    response_model=InventorySummaryResponse,
    tags=["inventory"],
    summary="Resumen por categoria de una sesion de inventario.",
)
def inventory_summary(
    session_id: int,
    repository: InventoryRepository = Depends(get_inventory_repository),
) -> InventorySummaryResponse:
    return InventorySummaryResponse(**repository.summary(session_id))


@router.post(
    "/inventory/product-stock-counts",
    response_model=ProductStockCountResponse,
    status_code=201,
    tags=["inventory"],
    summary="Confirma un conteo de stock de producto con evidencia de varias fotos.",
)
def create_product_stock_count(
    payload: ProductStockCountCreateRequest,
    repository: InventoryRepository = Depends(get_inventory_repository),
) -> ProductStockCountResponse:
    accepted_count = sum(1 for photo in payload.photos if photo.accepted)
    if accepted_count < 1:
        raise InvalidImageError("Debes confirmar al menos una foto valida para el conteo.")
    if payload.cantidad_final != accepted_count:
        raise InvalidImageError("La cantidad final debe coincidir con las fotos aceptadas.")
    record = repository.create_product_stock_count(payload.model_dump())
    return _product_stock_count_to_response(record)


@router.post(
    "/products/recognize",
    response_model=ProductRecognitionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Imagen inválida o demasiado grande."},
        503: {"model": ErrorResponse, "description": "Modelo YOLO/OCR no disponible."},
        500: {"model": ErrorResponse, "description": "Error interno de procesamiento."},
    },
    tags=["products"],
    summary="Reconoce datos iniciales de un producto retail peruano desde una imagen.",
)
async def recognize_product(
    image: UploadFile = File(..., description="Imagen JPG, PNG o WEBP del producto."),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
    settings: Settings = Depends(get_settings),
    pipeline: ProductRecognitionPipeline = Depends(get_product_pipeline),
    recognitions: RecognitionRepository = Depends(get_recognition_repository),
) -> ProductRecognitionResponse:
    trace_id = x_trace_id or str(uuid4())
    content_type = (image.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise InvalidImageError("El archivo enviado debe ser una imagen.")

    image_bytes = await image.read()
    max_bytes = settings.max_image_mb * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise InvalidImageError(f"La imagen supera el límite de {settings.max_image_mb} MB.")

    result = pipeline.process(image_bytes=image_bytes, trace_id=trace_id, source_name=image.filename)
    event = recognitions.create_from_response(
        image_bytes=image_bytes,
        content_type=image.content_type,
        source_name=image.filename,
        response=result,
    )
    return ProductRecognitionResponse(
        trace_id=result.trace_id,
        producto=result.producto,
        deteccion=result.deteccion,
        ocr=result.ocr,
        warnings=result.warnings,
        processing_ms=result.processing_ms,
        recognition_event_id=event.id,
        image_url=f"/api/v1/admin/reconocimientos/{event.id}/image",
    )


@router.get(
    "/admin/reconocimientos/stats",
    response_model=RecognitionStatsResponse,
    tags=["admin"],
    dependencies=[Depends(require_admin)],
    summary="Métricas de revisión de reconocimientos.",
)
def recognition_stats(
    repository: RecognitionRepository = Depends(get_recognition_repository),
) -> RecognitionStatsResponse:
    return RecognitionStatsResponse(**repository.stats())


@router.get(
    "/admin/reconocimientos",
    response_model=RecognitionEventsResponse,
    tags=["admin"],
    dependencies=[Depends(require_admin)],
    summary="Lista reconocimientos capturados para revisión asistida.",
)
def list_recognitions(
    status: str | None = Query(default=None, max_length=40),
    q: str | None = Query(default=None, max_length=120),
    category: str | None = Query(default=None, max_length=120),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repository: RecognitionRepository = Depends(get_recognition_repository),
    categorizer: ProductCategorizer = Depends(get_product_categorizer),
) -> RecognitionEventsResponse:
    records = repository.list(
        status=status,
        q=q,
        category=category,
        min_confidence=min_confidence,
        limit=limit,
        offset=offset,
    )
    return RecognitionEventsResponse(
        items=[_recognition_to_response(record, categorizer) for record in records]
    )


@router.get(
    "/admin/reconocimientos/{event_id}",
    response_model=RecognitionEventResponse,
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)
def get_recognition(
    event_id: int,
    repository: RecognitionRepository = Depends(get_recognition_repository),
    categorizer: ProductCategorizer = Depends(get_product_categorizer),
) -> RecognitionEventResponse:
    return _recognition_to_response(repository.get(event_id), categorizer)


@router.delete(
    "/admin/reconocimientos/{event_id}",
    status_code=204,
    tags=["admin"],
    dependencies=[Depends(require_admin)],
    summary="Elimina un reconocimiento capturado del panel de administración.",
)
def delete_recognition(
    event_id: int,
    repository: RecognitionRepository = Depends(get_recognition_repository),
) -> Response:
    repository.delete(event_id)
    return Response(status_code=204)


@router.get(
    "/admin/reconocimientos/{event_id}/image",
    tags=["admin"],
    include_in_schema=False,
    dependencies=[Depends(require_admin)],
)
def get_recognition_image(
    event_id: int,
    repository: RecognitionRepository = Depends(get_recognition_repository),
) -> Response:
    image, content_type = repository.get_image(event_id)
    return Response(content=image, media_type=content_type or "image/jpeg")


@router.put(
    "/admin/reconocimientos/{event_id}/review",
    response_model=RecognitionEventResponse,
    tags=["admin"],
    dependencies=[Depends(require_admin)],
    summary="Guarda validación, corrección o rechazo de un reconocimiento.",
)
def review_recognition(
    event_id: int,
    payload: RecognitionReviewRequest,
    categorizer: ProductCategorizer = Depends(get_product_categorizer),
    repository: RecognitionRepository = Depends(get_recognition_repository),
) -> RecognitionEventResponse:
    data = payload.model_dump()
    existing = repository.get(event_id)
    name = data.get("final_nombre_producto") or existing.predicted_nombre_producto
    if name and not data.get("final_categoria_sugerida"):
        categorized = categorizer.categorize(name, context_text=existing.ocr_text)
        data["final_categoria_sugerida"] = categorized.categoria_sugerida
        data["final_marca"] = data.get("final_marca") or categorized.marca
        data["final_tipo_producto"] = data.get("final_tipo_producto") or categorized.tipo_producto
        data["final_presentacion"] = data.get("final_presentacion") or categorized.presentacion
        data["final_contenido_neto"] = data.get("final_contenido_neto") or categorized.contenido_neto
        data["final_unidad_medida"] = data.get("final_unidad_medida") or categorized.unidad_medida
    data["use_for_training"] = 1 if payload.use_for_training else 0
    return _recognition_to_response(repository.review(event_id, data), categorizer)


@router.get(
    "/productos/suggestions",
    response_model=ProductSuggestionsResponse,
    tags=["productos"],
    summary="Sugerencias de nombre de producto (≤3) sobre productos guardados con fallback al catálogo.",
)
def suggest_products(
    q: str = Query(..., min_length=0, max_length=120),
    limit: int = Query(default=3, ge=1, le=10),
    context: str | None = Query(default=None, max_length=2000),
    source_name: str | None = Query(default=None, max_length=200),
    prominent_text: str | None = Query(default=None, max_length=200),
    service: ProductSuggestionService = Depends(get_suggestion_service),
) -> ProductSuggestionsResponse:
    items = service.suggest(
        q,
        limit=limit,
        context_text=context,
        source_name=source_name,
        prominent_text=prominent_text,
    )
    return ProductSuggestionsResponse(
        items=[
            ProductSuggestionItemSchema(
                nombre_producto=item.nombre_producto,
                marca=item.marca,
                tipo_producto=item.tipo_producto,
                categoria_sugerida=item.categoria_sugerida,
                source=item.source,
                product_id=item.product_id,
            )
            for item in items
        ]
    )


@router.post(
    "/productos/categorize",
    response_model=ProductCategorizeResponse,
    tags=["productos"],
    summary="Sugiere categoria y metadatos de producto desde el nombre.",
)
def categorize_product(
    payload: ProductCategorizeRequest,
    categorizer: ProductCategorizer = Depends(get_product_categorizer),
) -> ProductCategorizeResponse:
    result = categorizer.categorize(payload.nombre_producto, context_text=payload.context)
    return ProductCategorizeResponse(
        nombre_producto=result.nombre_producto,
        marca=result.marca,
        tipo_producto=result.tipo_producto,
        presentacion=result.presentacion,
        contenido_neto=result.contenido_neto,
        unidad_medida=result.unidad_medida,
        categoria_sugerida=result.categoria_sugerida,
        warnings=result.warnings or [],
    )


@router.get(
    "/productos",
    response_model=ProductListResponse,
    tags=["productos"],
    summary="Lista de productos registrados.",
)
def list_products(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repository: ProductRepository = Depends(get_product_repository),
) -> ProductListResponse:
    records = repository.list_all(limit=limit, offset=offset)
    return ProductListResponse(items=[_to_response(record) for record in records])


@router.get(
    "/productos/{product_id}",
    response_model=ProductResponse,
    tags=["productos"],
    responses={404: {"model": ErrorResponse}},
)
def get_product(
    product_id: int,
    repository: ProductRepository = Depends(get_product_repository),
) -> ProductResponse:
    return _to_response(repository.get(product_id))


@router.post(
    "/productos",
    response_model=ProductResponse,
    status_code=201,
    tags=["productos"],
    responses={409: {"model": ErrorResponse, "description": "Código de barras duplicado."}},
    summary="Registra un nuevo producto.",
)
def create_product(
    payload: ProductCreateRequest,
    repository: ProductRepository = Depends(get_product_repository),
    categorizer: ProductCategorizer = Depends(get_product_categorizer),
) -> ProductResponse:
    data = categorizer.enrich_payload(payload.model_dump())
    return _to_response(repository.create(data))


@router.put(
    "/productos/{product_id}",
    response_model=ProductResponse,
    tags=["productos"],
    responses={
        404: {"model": ErrorResponse, "description": "Producto no encontrado."},
        409: {"model": ErrorResponse, "description": "Código de barras duplicado."},
    },
    summary="Actualiza un producto existente.",
)
def update_product(
    product_id: int,
    payload: ProductUpdateRequest,
    repository: ProductRepository = Depends(get_product_repository),
    categorizer: ProductCategorizer = Depends(get_product_categorizer),
) -> ProductResponse:
    data = categorizer.enrich_payload(payload.model_dump())
    return _to_response(repository.update(product_id, data))
