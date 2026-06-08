from functools import lru_cache

from app.core.config import get_settings
from app.repositories.inventory import InventoryRepository
from app.repositories.products import ProductRepository
from app.repositories.recognitions import RecognitionRepository
from app.services.categorizer import ProductCategorizer, build_categorizer
from app.services.pipeline import ProductRecognitionPipeline, build_pipeline
from app.services.suggestions import ProductSuggestionService, build_suggestion_service


@lru_cache
def _cached_pipeline() -> ProductRecognitionPipeline:
    return build_pipeline(get_settings())


@lru_cache
def _cached_repository() -> ProductRepository:
    return ProductRepository(get_settings())


@lru_cache
def _cached_recognition_repository() -> RecognitionRepository:
    return RecognitionRepository(get_settings())


@lru_cache
def _cached_inventory_repository() -> InventoryRepository:
    return InventoryRepository(get_settings())


@lru_cache
def _cached_suggestion_service() -> ProductSuggestionService:
    return build_suggestion_service(_cached_repository())


@lru_cache
def _cached_categorizer() -> ProductCategorizer:
    return build_categorizer()


def get_product_pipeline() -> ProductRecognitionPipeline:
    return _cached_pipeline()


def get_product_repository() -> ProductRepository:
    return _cached_repository()


def get_recognition_repository() -> RecognitionRepository:
    return _cached_recognition_repository()


def get_inventory_repository() -> InventoryRepository:
    return _cached_inventory_repository()


def get_suggestion_service() -> ProductSuggestionService:
    return _cached_suggestion_service()


def get_product_categorizer() -> ProductCategorizer:
    return _cached_categorizer()
