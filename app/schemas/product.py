from pydantic import BaseModel, Field, field_validator


class ErrorResponse(BaseModel):
    trace_id: str | None = None
    error_code: str
    message: str
    detail: str | None = None


class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class DetectionMetadata(BaseModel):
    model: str
    bbox: BoundingBox
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    class_id: int | None = None
    class_name: str | None = None
    used_full_image_fallback: bool = False


class OcrLine(BaseModel):
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox_area: float | None = Field(default=None, ge=0.0)


class OcrMetadata(BaseModel):
    engine: str
    text: str
    prominent_text: str | None = None
    average_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    lines: list[OcrLine] = Field(default_factory=list)


class ProductSuggestion(BaseModel):
    nombre_producto: str | None = None
    marca: str | None = None
    tipo_producto: str | None = None
    presentacion: str | None = None
    contenido_neto: str | None = None
    unidad_medida: str | None = None
    categoria_sugerida: str | None = None


class ProductRecognitionResponse(BaseModel):
    trace_id: str
    producto: ProductSuggestion
    deteccion: DetectionMetadata
    ocr: OcrMetadata
    warnings: list[str] = Field(default_factory=list)
    processing_ms: int


class ProductSuggestionItemSchema(BaseModel):
    nombre_producto: str
    marca: str | None = None
    tipo_producto: str | None = None
    categoria_sugerida: str | None = None
    source: str
    product_id: int | None = None


class ProductSuggestionsResponse(BaseModel):
    items: list[ProductSuggestionItemSchema] = Field(default_factory=list)


class ProductCategorizeRequest(BaseModel):
    nombre_producto: str = Field(..., min_length=1, max_length=200)
    context: str | None = Field(default=None, max_length=2000)

    @field_validator("nombre_producto")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Campo obligatorio.")
        return cleaned


class ProductCategorizeResponse(ProductSuggestion):
    warnings: list[str] = Field(default_factory=list)


class ProductWriteBase(BaseModel):
    nombre_producto: str = Field(..., min_length=1, max_length=200)
    marca: str | None = Field(default=None, max_length=120)
    tipo_producto: str | None = Field(default=None, max_length=120)
    presentacion: str | None = Field(default=None, max_length=120)
    contenido_neto: str | None = Field(default=None, max_length=60)
    unidad_medida: str | None = Field(default=None, max_length=20)
    categoria_sugerida: str | None = Field(default=None, max_length=120)
    codigo_barras: str | None = Field(default=None, max_length=64)
    precio_venta: float = Field(..., ge=0)

    @field_validator("nombre_producto")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Campo obligatorio.")
        return cleaned

    @field_validator("categoria_sugerida", "codigo_barras")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("precio_venta")
    @classmethod
    def _round_price(cls, value: float) -> float:
        return round(float(value), 2)


class ProductCreateRequest(ProductWriteBase):
    pass


class ProductUpdateRequest(ProductWriteBase):
    pass


class ProductResponse(BaseModel):
    id: int
    nombre_producto: str
    marca: str | None = None
    tipo_producto: str | None = None
    presentacion: str | None = None
    contenido_neto: str | None = None
    unidad_medida: str | None = None
    categoria_sugerida: str | None = None
    codigo_barras: str | None = None
    precio_venta: float
    created_at: str
    updated_at: str


class ProductListResponse(BaseModel):
    items: list[ProductResponse] = Field(default_factory=list)


class RecognitionReviewRequest(BaseModel):
    status: str = Field(..., max_length=40)
    final_nombre_producto: str | None = Field(default=None, max_length=200)
    final_marca: str | None = Field(default=None, max_length=120)
    final_tipo_producto: str | None = Field(default=None, max_length=120)
    final_presentacion: str | None = Field(default=None, max_length=120)
    final_contenido_neto: str | None = Field(default=None, max_length=60)
    final_unidad_medida: str | None = Field(default=None, max_length=20)
    final_categoria_sugerida: str | None = Field(default=None, max_length=120)
    final_codigo_barras: str | None = Field(default=None, max_length=64)
    failure_reason: str | None = Field(default=None, max_length=80)
    review_notes: str | None = Field(default=None, max_length=500)
    use_for_training: bool = False
    linked_product_id: int | None = None


class RecognitionEventResponse(BaseModel):
    id: int
    trace_id: str
    source_name: str | None = None
    image_url: str
    status: str
    predicted_nombre_producto: str | None = None
    predicted_marca: str | None = None
    predicted_tipo_producto: str | None = None
    predicted_presentacion: str | None = None
    predicted_contenido_neto: str | None = None
    predicted_unidad_medida: str | None = None
    predicted_categoria_sugerida: str | None = None
    final_nombre_producto: str | None = None
    final_marca: str | None = None
    final_tipo_producto: str | None = None
    final_presentacion: str | None = None
    final_contenido_neto: str | None = None
    final_unidad_medida: str | None = None
    final_categoria_sugerida: str | None = None
    final_codigo_barras: str | None = None
    yolo_confidence: float | None = None
    yolo_class_name: str | None = None
    ocr_confidence: float | None = None
    ocr_text: str | None = None
    warnings: list[str] = Field(default_factory=list)
    bbox: dict[str, int] | None = None
    failure_reason: str | None = None
    review_notes: str | None = None
    use_for_training: bool = False
    linked_product_id: int | None = None
    recognition: dict = Field(default_factory=dict)
    reviewed_at: str | None = None
    created_at: str
    updated_at: str


class RecognitionEventsResponse(BaseModel):
    items: list[RecognitionEventResponse] = Field(default_factory=list)


class RecognitionStatsResponse(BaseModel):
    pending_review: int
    validated: int
    corrected: int
    rejected: int
    training_candidates: int
    total: int
    precision: float
