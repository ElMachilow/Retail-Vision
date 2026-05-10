from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.core.config import Settings
from app.core.exceptions import ModelUnavailableError, ProcessingError


@dataclass(frozen=True)
class RegionDetection:
    bbox: tuple[int, int, int, int]
    model: str
    confidence: float | None
    class_id: int | None
    class_name: str | None
    used_full_image_fallback: bool = False


class ProductRegionDetector(Protocol):
    def detect(self, image: np.ndarray) -> RegionDetection:
        ...


class YoloRegionDetector:
    """YOLO-based detector for the product/package region.

    The MVP can start with YOLOv8n to prove the pipeline. For thesis validation,
    configure ``YOLO_MODEL_PATH`` with weights fine-tuned on Peruvian product
    packages or label regions.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except Exception as exc:  # pragma: no cover - depends on local install
                raise ModelUnavailableError(
                    "YOLO no está disponible.",
                    "Instala ultralytics o revisa el entorno Docker/local.",
                ) from exc
            try:
                self._model = YOLO(self.settings.yolo_model_path)
            except Exception as exc:  # pragma: no cover - depends on model path/download
                raise ModelUnavailableError(
                    "No se pudo cargar el modelo YOLO.",
                    f"Modelo configurado: {self.settings.yolo_model_path}",
                ) from exc
        return self._model

    def detect(self, image: np.ndarray) -> RegionDetection:
        try:
            results = self.model.predict(
                source=image,
                conf=self.settings.yolo_confidence_threshold,
                device=self.settings.yolo_device,
                verbose=False,
            )
        except ModelUnavailableError:
            raise
        except Exception as exc:  # pragma: no cover - depends on ultralytics runtime
            raise ProcessingError("Error ejecutando la detección YOLO.") from exc

        if not results:
            return self._fallback_or_fail(image)

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return self._fallback_or_fail(image)

        names = getattr(result, "names", {}) or {}
        if self._is_roboflow_label_model(names):
            xyxy, confidence, class_id, class_name = self._select_roboflow_roi(boxes, names)
        else:
            best = self._select_best_box(boxes)
            xyxy = best.xyxy[0].detach().cpu().numpy().astype(int).tolist()
            confidence = float(best.conf[0].detach().cpu().item()) if best.conf is not None else None
            class_id = int(best.cls[0].detach().cpu().item()) if best.cls is not None else None
            class_name = names.get(class_id) if class_id is not None else None

        return RegionDetection(
            bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
            model=self.settings.yolo_model_path,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
        )

    def _is_roboflow_label_model(self, names: dict) -> bool:
        class_names = {str(name) for name in names.values()}
        return {"brand", "product_name", "net_weight"}.issubset(class_names)

    def _select_roboflow_roi(self, boxes, names: dict):
        selected = []
        class_ids: set[int] = set()
        confidences: list[float] = []
        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
            selected.append(xyxy)
            if box.conf is not None:
                confidences.append(float(box.conf[0].detach().cpu().item()))
            if box.cls is not None:
                class_ids.add(int(box.cls[0].detach().cpu().item()))

        x_min = min(item[0] for item in selected)
        y_min = min(item[1] for item in selected)
        x_max = max(item[2] for item in selected)
        y_max = max(item[3] for item in selected)
        class_name = "+".join(
            str(names[class_id]) for class_id in sorted(class_ids) if class_id in names
        )
        return [x_min, y_min, x_max, y_max], max(confidences) if confidences else None, None, class_name or None

    def _select_best_box(self, boxes):
        scored = []
        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()
            width = max(1.0, float(xyxy[2] - xyxy[0]))
            height = max(1.0, float(xyxy[3] - xyxy[1]))
            confidence = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
            scored.append((width * height * confidence, box))
        return max(scored, key=lambda item: item[0])[1]

    def _fallback_or_fail(self, image: np.ndarray) -> RegionDetection:
        if not self.settings.allow_full_image_fallback:
            raise ProcessingError("YOLO no detectó una región relevante del producto.")

        height, width = image.shape[:2]
        return RegionDetection(
            bbox=(0, 0, width, height),
            model=self.settings.yolo_model_path,
            confidence=None,
            class_id=None,
            class_name=None,
            used_full_image_fallback=True,
        )

