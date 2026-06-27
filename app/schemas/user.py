from pydantic import BaseModel, EmailStr, Field


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=80)
    password: str = Field(..., min_length=8, max_length=128)
    display_name: str | None = Field(default=None, max_length=120)


class UserResponse(BaseModel):
    id: int
    username: str
    display_name: str | None = None
    role: str
    created_at: str
    updated_at: str


class UserLoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=80)
    password: str = Field(..., min_length=8, max_length=128)


class RecognitionReviewByUserResponse(BaseModel):
    id: int
    user_id: int
    username: str
    recognition_event_id: int
    status: str
    final_nombre_producto: str | None = None
    final_marca: str | None = None
    final_tipo_producto: str | None = None
    final_presentacion: str | None = None
    final_contenido_neto: str | None = None
    final_unidad_medida: str | None = None
    final_categoria_sugerida: str | None = None
    final_codigo_barras: str | None = None
    failure_reason: str | None = None
    review_notes: str | None = None
    use_for_training: bool = False
    linked_product_id: int | None = None
    reviewed_at: str | None = None
    created_at: str
    updated_at: str
