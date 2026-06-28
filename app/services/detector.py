from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.core.config import Settings
from app.core.exceptions import ModelUnavailableError, ProcessingError


@dataclass(frozen=True)
class FieldDetection:
    bbox: tuple[int, int, int, int]
    confidence: float | None
    class_id: int | None
    class_name: str | None


@dataclass(frozen=True)
class RegionDetection:
    bbox: tuple[int, int, int, int]
    model: str
    confidence: float | None
    class_id: int | None
    class_name: str | None
    used_full_image_fallback: bool = False
    fields: tuple[FieldDetection, ...] = ()


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
                imgsz=self.settings.yolo_imgsz,
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
        fields: tuple[FieldDetection, ...] = ()
        if self._is_roboflow_label_model(names):
            xyxy, confidence, class_id, class_name, fields = self._select_roboflow_roi(boxes, names)
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
            fields=fields,
        )

    def _is_roboflow_label_model(self, names: dict) -> bool:
        class_names = {str(name) for name in names.values()}
        return {"brand", "product_name", "net_weight"}.issubset(class_names)

    def _select_roboflow_roi(self, boxes, names: dict):
        selected_fields = self._select_roboflow_fields(boxes, names)
        selected = [list(field.bbox) for field in selected_fields]
        class_ids: set[int] = set()
        confidences: list[float] = []
        for field in selected_fields:
            if field.confidence is not None:
                confidences.append(field.confidence)
            if field.class_id is not None:
                class_id = field.class_id
                class_ids.add(class_id)

        x_min = min(item[0] for item in selected)
        y_min = min(item[1] for item in selected)
        x_max = max(item[2] for item in selected)
        y_max = max(item[3] for item in selected)
        class_name = "+".join(
            str(names[class_id]) for class_id in sorted(class_ids) if class_id in names
        )
        fields = tuple(sorted(selected_fields, key=self._field_sort_key))
        return [x_min, y_min, x_max, y_max], max(confidences) if confidences else None, None, class_name or None, fields

    def _select_roboflow_fields(self, boxes, names: dict) -> list[FieldDetection]:
        expected_classes = {"brand", "product_name", "net_weight", "extra_detail"}
        per_class_limits = {
            "brand": 1,
            "product_name": 1,
            "net_weight": 1,
            "extra_detail": 2,
        }
        candidates_by_class: dict[str, list[FieldDetection]] = {
            class_name: [] for class_name in expected_classes
        }

        for box in boxes:
            class_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else None
            class_name = names.get(class_id) if class_id is not None else None
            if class_name not in expected_classes:
                continue

            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
            confidence = (
                float(box.conf[0].detach().cpu().item())
                if box.conf is not None
                else None
            )
            candidates_by_class[class_name].append(
                FieldDetection(
                    bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                )
            )

        selected: list[FieldDetection] = []
        for class_name, candidates in candidates_by_class.items():
            limit = per_class_limits[class_name]
            candidates.sort(key=lambda item: item.confidence or 0.0, reverse=True)
            selected.extend(candidates[:limit])

        if selected:
            return selected

        return [self._field_from_box(box, names) for box in boxes[:1]]

    def _field_from_box(self, box, names: dict) -> FieldDetection:
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
        confidence = (
            float(box.conf[0].detach().cpu().item())
            if box.conf is not None
            else None
        )
        class_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else None
        return FieldDetection(
            bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
            confidence=confidence,
            class_id=class_id,
            class_name=names.get(class_id) if class_id is not None else None,
        )

    def _field_sort_key(self, field: FieldDetection) -> tuple[int, int, int]:
        priority = {
            "brand": 0,
            "product_name": 1,
            "extra_detail": 2,
            "net_weight": 3,
        }.get(field.class_name or "", 9)
        x_min, y_min, _, _ = field.bbox
        return priority, y_min, x_min

    def _box_area(self, box) -> float:
        xyxy = box.xyxy[0].detach().cpu().numpy()
        width = max(0.0, float(xyxy[2] - xyxy[0]))
        height = max(0.0, float(xyxy[3] - xyxy[1]))
        return width * height

    def _select_best_box(self, boxes):
        scored = []
        for box in boxes:
            area = self._box_area(box)
            confidence = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
            score = area * (confidence ** 1.25)
            scored.append((score, box))
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

