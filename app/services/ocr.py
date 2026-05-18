from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.core.config import Settings
from app.core.exceptions import ModelUnavailableError, ProcessingError


@dataclass(frozen=True)
class OcrTextLine:
    text: str
    confidence: float | None
    bbox_area: float | None = None


@dataclass(frozen=True)
class OcrResult:
    engine: str
    text: str
    average_confidence: float | None
    lines: list[OcrTextLine]


class TextExtractor(Protocol):
    def extract(self, image: np.ndarray) -> OcrResult:
        ...


class PaddleOcrTextExtractor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from paddleocr import PaddleOCR
            except Exception as exc:  # pragma: no cover - depends on local install
                raise ModelUnavailableError(
                    "PaddleOCR no está disponible.",
                    "Instala paddleocr y paddlepaddle o usa la imagen Docker.",
                ) from exc

            try:
                self._client = self._build_client(PaddleOCR)
            except Exception as exc:  # pragma: no cover - depends on model downloads
                raise ModelUnavailableError(
                    "No se pudo inicializar PaddleOCR.",
                    f"Idioma configurado: {self.settings.paddle_ocr_lang}",
                ) from exc
        return self._client

    def _build_client(self, paddle_ocr_class):
        common_kwargs = {"lang": self.settings.paddle_ocr_lang}
        v3_kwargs = {
            **common_kwargs,
            "use_textline_orientation": self.settings.paddle_use_angle_cls,
        }
        v2_kwargs = {
            **common_kwargs,
            "use_angle_cls": self.settings.paddle_use_angle_cls,
        }

        try:
            return paddle_ocr_class(**v3_kwargs)
        except (TypeError, ValueError):
            return paddle_ocr_class(**v2_kwargs)

    def extract(self, image: np.ndarray) -> OcrResult:
        try:
            raw_result = self._run_ocr(image)
        except ModelUnavailableError:
            raise
        except Exception as exc:  # pragma: no cover - depends on OCR runtime
            raise ProcessingError("Error ejecutando OCR.") from exc

        lines = self._parse_lines(raw_result)
        text = "\n".join(line.text for line in lines)
        confidences = [line.confidence for line in lines if line.confidence is not None]
        average = sum(confidences) / len(confidences) if confidences else None
        return OcrResult(engine="paddleocr", text=text, average_confidence=average, lines=lines)

    def _run_ocr(self, image: np.ndarray):
        client = self.client
        if hasattr(client, "ocr"):
            try:
                return client.ocr(image, cls=self.settings.paddle_use_angle_cls)
            except TypeError:
                return client.ocr(image)
        if hasattr(client, "predict"):
            return client.predict(image)
        raise ProcessingError("La instancia de PaddleOCR no expone métodos ocr/predict compatibles.")

    def _parse_lines(self, raw_result) -> list[OcrTextLine]:
        if not raw_result:
            return []

        dict_lines = self._parse_dict_lines(raw_result)
        if dict_lines:
            return dict_lines

        if len(raw_result) == 1 and isinstance(raw_result[0], list):
            candidates = raw_result[0]
        else:
            candidates = raw_result

        lines: list[OcrTextLine] = []
        for item in candidates:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            payload = item[1]
            if not isinstance(payload, (list, tuple)) or not payload:
                continue
            text = str(payload[0]).strip()
            if not text:
                continue
            confidence = None
            if len(payload) > 1:
                try:
                    confidence = float(payload[1])
                except (TypeError, ValueError):
                    confidence = None
            lines.append(
                OcrTextLine(
                    text=text,
                    confidence=confidence,
                    bbox_area=self._box_area(item[0] if item else None),
                )
            )
        return lines

    def _parse_dict_lines(self, raw_result) -> list[OcrTextLine]:
        lines: list[OcrTextLine] = []
        candidates = raw_result if isinstance(raw_result, list) else [raw_result]
        for item in candidates:
            if hasattr(item, "json"):
                try:
                    item = item.json
                except TypeError:
                    item = item.json()
            if not isinstance(item, dict):
                continue

            payload = item.get("res") if isinstance(item.get("res"), dict) else item
            texts = payload.get("rec_texts") or payload.get("texts") or []
            scores = payload.get("rec_scores") or payload.get("scores") or []
            boxes = self._first_present(
                payload.get("rec_boxes"),
                payload.get("dt_boxes"),
                payload.get("rec_polys"),
                payload.get("dt_polys"),
                payload.get("boxes"),
            )
            for index, text in enumerate(texts):
                clean_text = str(text).strip()
                if not clean_text:
                    continue
                confidence = None
                if index < len(scores):
                    try:
                        confidence = float(scores[index])
                    except (TypeError, ValueError):
                        confidence = None
                box = boxes[index] if index < len(boxes) else None
                lines.append(OcrTextLine(text=clean_text, confidence=confidence, bbox_area=self._box_area(box)))
        return lines

    def _first_present(self, *values):
        for value in values:
            if value is None:
                continue
            try:
                if len(value) == 0:
                    continue
            except TypeError:
                pass
            return value
        return []

    def _box_area(self, box) -> float | None:
        if box is None:
            return None
        try:
            array = np.asarray(box, dtype=float)
        except (TypeError, ValueError):
            return None
        if array.size < 4:
            return None
        if array.ndim == 1 and array.size >= 4:
            x_min, y_min, x_max, y_max = array[:4]
            return max(0.0, float(x_max - x_min)) * max(0.0, float(y_max - y_min))
        points = array.reshape(-1, 2)
        x_min = float(points[:, 0].min())
        x_max = float(points[:, 0].max())
        y_min = float(points[:, 1].min())
        y_max = float(points[:, 1].max())
        return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
