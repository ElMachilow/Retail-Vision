from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Tesis Retail Vision MVP"
    app_env: str = "local"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    cors_allow_origins: str = "*"
    expose_internal_errors: bool = False
    max_image_mb: int = Field(default=10, ge=1, le=50)
    admin_username: str = "admin"
    admin_password: str = "admin123"
    admin_session_secret: str = "change-this-admin-session-secret"
    admin_session_max_age_seconds: int = Field(default=8 * 60 * 60, ge=300, le=7 * 24 * 60 * 60)

    yolo_model_path: str = "yolov8n.pt"
    yolo_confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    yolo_device: str = "cpu"
    yolo_imgsz: int = Field(default=416, ge=160, le=1280)
    allow_full_image_fallback: bool = True

    ocr_engine: str = "paddle"
    paddle_ocr_lang: str = "es"
    paddle_use_angle_cls: bool = True

    persist_debug_images: bool = False
    debug_image_dir: Path = Path("runtime/debug")

    db_backend: str = "sqlite"
    sqlite_path: Path = Path("runtime/products.db")
    mysql_host: str = "127.0.0.1"
    mysql_port: int = Field(default=3306, ge=1, le=65535)
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "retail_vision"
    mysql_charset: str = "utf8mb4"
    recognition_image_dir: Path = Path("runtime/recognition-images")


@lru_cache
def get_settings() -> Settings:
    return Settings()

