import re
import time
from dataclasses import dataclass

from app.core.config import Settings
from app.core.exceptions import ProcessingError
from app.core.logging import get_trace_logger
from app.schemas.product import (
    BoundingBox,
    DetectionMetadata,
    OcrLine,
    OcrMetadata,
    ProductRecognitionResponse,
    ProductSuggestion,
)
from app.services.detector import FieldDetection, ProductRegionDetector, RegionDetection, YoloRegionDetector
from app.services.image_utils import (
    crop_image,
    decode_image,
    prepare_for_ocr,
    save_debug_image,
    validate_image_quality,
)
from app.services.normalizer import ProductTextNormalizer
from app.services.ocr import OcrResult, OcrTextLine, PaddleOcrTextExtractor, TextExtractor


@dataclass
class ProductRecognitionPipeline:
    settings: Settings
    detector: ProductRegionDetector
    ocr: TextExtractor
    normalizer: ProductTextNormalizer

    def process(
        self,
        image_bytes: bytes,
        trace_id: str,
        source_name: str | None = None,
    ) -> ProductRecognitionResponse:
        logger = get_trace_logger(__name__, trace_id)
        start = time.perf_counter()
        logger.info("inicio procesamiento producto")

        image = decode_image(image_bytes)
        logger.info("imagen decodificada")
        quality = validate_image_quality(image)
        logger.info(
            "calidad imagen laplacian=%s edge_density=%s contrast=%s detail_score=%s blur_percentage=%s",
            round(quality.laplacian_variance, 2),
            round(quality.edge_density, 4),
            round(quality.contrast, 2),
            round(quality.detail_score, 2),
            round(quality.blur_percentage, 1),
        )

        detection = self.detector.detect(image)
        logger.info(
            "deteccion yolo completada bbox=%s confidence=%s fallback=%s",
            detection.bbox,
            detection.confidence,
            detection.used_full_image_fallback,
        )

        ocr_result = self._extract_ocr(image, detection, trace_id)
        logger.info("ocr completado lineas=%s", len(ocr_result.lines))

        prominent_text = self._prominent_ocr_text(ocr_result)
        normalized = self.normalizer.normalize(
            ocr_result.text,
            source_name=source_name,
            prominent_text=prominent_text,
        )
        if detection.fields and self._needs_full_image_ocr(normalized):
            full_ocr_result = self._extract_full_image_ocr(image, trace_id)
            ocr_result = self._merge_ocr_results(ocr_result, full_ocr_result)
            prominent_text = self._prominent_ocr_text(ocr_result)
            normalized = self.normalizer.normalize(
                ocr_result.text,
                source_name=source_name,
                prominent_text=prominent_text,
            )
            logger.info("ocr imagen completa fusionado lineas=%s", len(ocr_result.lines))

        warnings = list(normalized.warnings)
        if detection.used_full_image_fallback:
            warnings.append("YOLO no detectó una región; se procesó la imagen completa como respaldo técnico.")
        if not ocr_result.lines:
            warnings.append("OCR sin líneas detectadas; el producto debe revisarse manualmente.")

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        if elapsed_ms < 0:  # pragma: no cover - defensive
            raise ProcessingError("Tiempo de procesamiento inválido.")

        return ProductRecognitionResponse(
            trace_id=trace_id,
            producto=ProductSuggestion(
                nombre_producto=normalized.nombre_producto,
                marca=normalized.marca,
                tipo_producto=normalized.tipo_producto,
                presentacion=normalized.presentacion,
                contenido_neto=normalized.contenido_neto,
                unidad_medida=normalized.unidad_medida,
                categoria_sugerida=normalized.categoria_sugerida,
            ),
            deteccion=DetectionMetadata(
                model=detection.model,
                bbox=BoundingBox(
                    x_min=detection.bbox[0],
                    y_min=detection.bbox[1],
                    x_max=detection.bbox[2],
                    y_max=detection.bbox[3],
                ),
                confidence=detection.confidence,
                class_id=detection.class_id,
                class_name=detection.class_name,
                used_full_image_fallback=detection.used_full_image_fallback,
            ),
            ocr=OcrMetadata(
                engine=ocr_result.engine,
                text=ocr_result.text,
                prominent_text=prominent_text,
                average_confidence=ocr_result.average_confidence,
                lines=[
                    OcrLine(text=line.text, confidence=line.confidence, bbox_area=line.bbox_area)
                    for line in ocr_result.lines
                ],
            ),
            warnings=warnings,
            processing_ms=elapsed_ms,
        )

    def _extract_ocr(
        self,
        image,
        detection: RegionDetection,
        trace_id: str,
    ) -> OcrResult:
        if detection.fields:
            return self._extract_field_aware_ocr(image, detection, trace_id)

        cropped = crop_image(image, detection.bbox)
        ocr_image = prepare_for_ocr(cropped)
        if self.settings.persist_debug_images:
            save_debug_image(ocr_image, self.settings.debug_image_dir, f"{trace_id}_roi.jpg")
        return self.ocr.extract(ocr_image)

    def _extract_field_aware_ocr(
        self,
        image,
        detection: RegionDetection,
        trace_id: str,
    ) -> OcrResult:
        all_lines: list[OcrTextLine] = []
        engines: list[str] = []
        seen_text: set[str] = set()

        for index, field in enumerate(detection.fields):
            field_result = self._extract_field_ocr(image, field, trace_id, index)
            engines.append(field_result.engine)
            for line in field_result.lines:
                self._append_unique_line(all_lines, seen_text, line)

        roi_result = self._extract_roi_ocr(image, detection, trace_id)
        engines.append(roi_result.engine)
        for line in roi_result.lines:
            self._append_unique_line(all_lines, seen_text, line)

        confidences = [line.confidence for line in all_lines if line.confidence is not None]
        average = sum(confidences) / len(confidences) if confidences else None
        engine = engines[0] if engines else "unknown"
        return OcrResult(
            engine=engine,
            text="\n".join(line.text for line in all_lines),
            average_confidence=average,
            lines=all_lines,
        )

    def _extract_field_ocr(
        self,
        image,
        field: FieldDetection,
        trace_id: str,
        index: int,
    ) -> OcrResult:
        cropped = crop_image(image, self._expand_bbox(image, field.bbox))
        ocr_image = prepare_for_ocr(cropped)
        if self.settings.persist_debug_images:
            class_name = field.class_name or "field"
            save_debug_image(ocr_image, self.settings.debug_image_dir, f"{trace_id}_{index}_{class_name}.jpg")
        return self.ocr.extract(ocr_image)

    def _extract_roi_ocr(
        self,
        image,
        detection: RegionDetection,
        trace_id: str,
    ) -> OcrResult:
        cropped = crop_image(image, self._expand_bbox(image, detection.bbox, margin_ratio=0.04))
        ocr_image = prepare_for_ocr(cropped)
        if self.settings.persist_debug_images:
            save_debug_image(ocr_image, self.settings.debug_image_dir, f"{trace_id}_roi.jpg")
        return self.ocr.extract(ocr_image)

    def _extract_full_image_ocr(self, image, trace_id: str) -> OcrResult:
        ocr_image = prepare_for_ocr(image)
        if self.settings.persist_debug_images:
            save_debug_image(ocr_image, self.settings.debug_image_dir, f"{trace_id}_full.jpg")
        return self.ocr.extract(ocr_image)

    def _needs_full_image_ocr(self, normalized) -> bool:
        return (
            normalized.nombre_producto is None
            or normalized.marca is None
            or normalized.tipo_producto is None
            or normalized.categoria_sugerida is None
        )

    def _merge_ocr_results(self, primary: OcrResult, secondary: OcrResult) -> OcrResult:
        lines: list[OcrTextLine] = []
        seen_text: set[str] = set()
        for result in (primary, secondary):
            for line in result.lines:
                self._append_unique_line(lines, seen_text, line)

        confidences = [line.confidence for line in lines if line.confidence is not None]
        average = sum(confidences) / len(confidences) if confidences else None
        return OcrResult(
            engine=primary.engine,
            text="\n".join(line.text for line in lines),
            average_confidence=average,
            lines=lines,
        )

    def _prominent_ocr_text(self, ocr_result: OcrResult) -> str | None:
        candidates: list[tuple[float, int, str]] = []
        for index, line in enumerate(ocr_result.lines):
            text = " ".join(line.text.split())
            if not self._looks_like_product_name_candidate(text):
                continue
            area = line.bbox_area or 0.0
            alpha_chars = len([char for char in text if char.isalpha()])
            if area <= 0:
                area = float(alpha_chars)
            candidates.append((area, -index, text))
        if not candidates:
            return None
        return max(candidates)[2]

    def _looks_like_product_name_candidate(self, text: str) -> bool:
        if len(text) < 3:
            return False
        folded = self.normalizer._fold(text)
        if re.fullmatch(r"[\d\s.,/%+-]+", folded):
            return False
        if re.search(r"\b\d+(?:[.,]\d+)?\s*(mg|mcg|g|kg|ml|l|%)\b", folded):
            return False
        if re.search(r"\b(via|oral|laboratorio|laboratorios|contenido|neto)\b", folded):
            return False
        return bool(re.search(r"[a-z]", folded))

    def _expand_bbox(
        self,
        image,
        bbox: tuple[int, int, int, int],
        margin_ratio: float = 0.08,
    ) -> tuple[int, int, int, int]:
        height, width = image.shape[:2]
        x_min, y_min, x_max, y_max = bbox
        box_width = max(1, x_max - x_min)
        box_height = max(1, y_max - y_min)
        margin = max(4, int(max(box_width, box_height) * margin_ratio))
        return (
            max(0, x_min - margin),
            max(0, y_min - margin),
            min(width, x_max + margin),
            min(height, y_max + margin),
        )

    def _append_unique_line(
        self,
        lines: list[OcrTextLine],
        seen_text: set[str],
        line: OcrTextLine,
    ) -> None:
        normalized = " ".join(line.text.lower().split())
        if not normalized or normalized in seen_text:
            return
        seen_text.add(normalized)
        lines.append(line)


def build_pipeline(settings: Settings) -> ProductRecognitionPipeline:
    if settings.ocr_engine.lower() != "paddle":
        raise ProcessingError(f"OCR_ENGINE no soportado: {settings.ocr_engine}")

    return ProductRecognitionPipeline(
        settings=settings,
        detector=YoloRegionDetector(settings),
        ocr=PaddleOcrTextExtractor(settings),
        normalizer=ProductTextNormalizer(),
    )
