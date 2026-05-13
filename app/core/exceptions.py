class AppError(Exception):
    """Base class for expected application errors."""

    status_code = 500
    error_code = "APP_ERROR"

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class InvalidImageError(AppError):
    status_code = 400
    error_code = "INVALID_IMAGE"


class BlurryImageError(InvalidImageError):
    error_code = "BLURRY_IMAGE"


class ModelUnavailableError(AppError):
    status_code = 503
    error_code = "MODEL_UNAVAILABLE"


class ProcessingError(AppError):
    status_code = 500
    error_code = "PROCESSING_ERROR"

