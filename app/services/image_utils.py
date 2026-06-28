from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np

from app.core.exceptions import BlurryImageError, InvalidImageError


@dataclass(frozen=True)
class ImageQualityReport:
    is_blurry: bool
    laplacian_variance: float
    edge_density: float
    contrast: float
    detail_score: float
    blur_percentage: float


def decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise InvalidImageError("La imagen enviada está vacía.")

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("No se pudo decodificar la imagen. Usa JPG, PNG o WEBP válido.")
    return image


def validate_image_quality(image: np.ndarray) -> ImageQualityReport:
    report = assess_image_quality(image)
    if report.is_blurry:
        raise BlurryImageError(
            "La foto se ve borrosa o desenfocada. Toma la foto nuevamente enfocando bien el producto.",
            detail=(
                "BLURRY_IMAGE "
                f"laplacian_variance={report.laplacian_variance:.2f} "
                f"edge_density={report.edge_density:.4f} "
                f"contrast={report.contrast:.2f} "
                f"detail_score={report.detail_score:.2f} "
                f"blur_percentage={report.blur_percentage:.1f}"
            ),
        )
    return report


def assess_image_quality(image: np.ndarray) -> ImageQualityReport:
    height, width = image.shape[:2]
    if height < 40 or width < 40:
        return ImageQualityReport(True, 0.0, 0.0, 0.0, 0.0, 100.0)

    # The product is normally centered in the mobile capture. Measuring the
    # central area avoids accepting a blurred product just because the
    # background monitor or desk has sharp edges.
    x_margin = int(width * 0.15)
    y_margin = int(height * 0.15)
    center = image[y_margin : height - y_margin, x_margin : width - x_margin]
    if center.size == 0:
        center = image

    max_side = max(center.shape[:2])
    if max_side > 900:
        scale = 900 / max_side
        center = cv2.resize(center, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    contrast = float(gray.std())

    # Weighted score tuned for mobile product photos: strong blur produces low
    # Laplacian and very low edge density in the label area, even when the
    # background contains some sharp structures.
    detail_score = laplacian_variance + (edge_density * 900.0) + (contrast * 0.8)
    sharpness_percentage = min(100.0, max(0.0, (detail_score / 160.0) * 100.0))
    blur_percentage = 100.0 - sharpness_percentage
    is_blurry = blur_percentage >= 90.0

    return ImageQualityReport(
        is_blurry=is_blurry,
        laplacian_variance=laplacian_variance,
        edge_density=edge_density,
        contrast=contrast,
        detail_score=detail_score,
        blur_percentage=blur_percentage,
    )


def crop_image(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, min(width - 1, x_min))
    x_max = max(1, min(width, x_max))
    y_min = max(0, min(height - 1, y_min))
    y_max = max(1, min(height, y_max))
    if x_max <= x_min or y_max <= y_min:
        raise InvalidImageError("La región detectada no es válida para recorte.")
    return image[y_min:y_max, x_min:x_max]


def prepare_for_ocr(image: np.ndarray, target_max_side: int = 800) -> np.ndarray:
    """Preprocess package crops for OCR while preserving text color cues."""

    image = trim_plain_background(image)
    image = deskew_image(image)
    height, width = image.shape[:2]
    max_side = max(height, width)
    if max_side < target_max_side:
        scale = target_max_side / max_side
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.45, blurred, -0.45, 0)
    return sharpened


def deskew_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    if coords.size < 10:
        return image

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle += 90
    if abs(angle) < 1.0:
        return image

    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def trim_plain_background(image: np.ndarray) -> np.ndarray:
    """Remove wide plain margins so OCR spends resolution on the package text."""

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    foreground = (saturation > 18) | (value < 242) | (gray < 238)
    foreground = foreground.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    crop_area = w * h
    image_area = width * height
    if crop_area < image_area * 0.03 or crop_area > image_area * 0.96:
        return image

    margin = max(8, int(max(width, height) * 0.04))
    x_min = max(0, x - margin)
    y_min = max(0, y - margin)
    x_max = min(width, x + w + margin)
    y_max = min(height, y + h + margin)
    return image[y_min:y_max, x_min:x_max]


def save_debug_image(image: np.ndarray, directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / filename
    cv2.imwrite(str(output_path), image)
    return output_path

