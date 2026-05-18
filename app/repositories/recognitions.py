import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.config import Settings
from app.core.exceptions import AppError
from app.db.connection import get_connection
from app.schemas.product import ProductRecognitionResponse


class RecognitionEventNotFoundError(AppError):
    status_code = 404
    error_code = "RECOGNITION_EVENT_NOT_FOUND"


class InvalidReviewStatusError(AppError):
    status_code = 400
    error_code = "INVALID_REVIEW_STATUS"


VALID_REVIEW_STATUSES = {
    "pending_review",
    "validated",
    "corrected",
    "rejected",
    "duplicate",
    "ignored",
    "training_candidate",
    "used_for_training",
}


@dataclass(frozen=True)
class RecognitionEventRecord:
    id: int
    trace_id: str
    source_name: str | None
    image_content_type: str | None
    status: str
    predicted_nombre_producto: str | None
    predicted_marca: str | None
    predicted_tipo_producto: str | None
    predicted_presentacion: str | None
    predicted_contenido_neto: str | None
    predicted_unidad_medida: str | None
    predicted_categoria_sugerida: str | None
    final_nombre_producto: str | None
    final_marca: str | None
    final_tipo_producto: str | None
    final_presentacion: str | None
    final_contenido_neto: str | None
    final_unidad_medida: str | None
    final_categoria_sugerida: str | None
    final_codigo_barras: str | None
    yolo_confidence: float | None
    yolo_class_name: str | None
    ocr_confidence: float | None
    ocr_text: str | None
    warnings: list[str]
    bbox: dict[str, int] | None
    failure_reason: str | None
    review_notes: str | None
    use_for_training: bool
    linked_product_id: int | None
    recognition: dict[str, Any]
    reviewed_at: str | None
    created_at: str
    updated_at: str


def _json_loads(value: str | None, fallback):
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _row_to_record(row: sqlite3.Row) -> RecognitionEventRecord:
    return RecognitionEventRecord(
        id=row["id"],
        trace_id=row["trace_id"],
        source_name=row["source_name"],
        image_content_type=row["image_content_type"],
        status=row["status"],
        predicted_nombre_producto=row["predicted_nombre_producto"],
        predicted_marca=row["predicted_marca"],
        predicted_tipo_producto=row["predicted_tipo_producto"],
        predicted_presentacion=row["predicted_presentacion"],
        predicted_contenido_neto=row["predicted_contenido_neto"],
        predicted_unidad_medida=row["predicted_unidad_medida"],
        predicted_categoria_sugerida=row["predicted_categoria_sugerida"],
        final_nombre_producto=row["final_nombre_producto"],
        final_marca=row["final_marca"],
        final_tipo_producto=row["final_tipo_producto"],
        final_presentacion=row["final_presentacion"],
        final_contenido_neto=row["final_contenido_neto"],
        final_unidad_medida=row["final_unidad_medida"],
        final_categoria_sugerida=row["final_categoria_sugerida"],
        final_codigo_barras=row["final_codigo_barras"],
        yolo_confidence=row["yolo_confidence"],
        yolo_class_name=row["yolo_class_name"],
        ocr_confidence=row["ocr_confidence"],
        ocr_text=row["ocr_text"],
        warnings=_json_loads(row["warnings_json"], []),
        bbox=_json_loads(row["bbox_json"], None),
        failure_reason=row["failure_reason"],
        review_notes=row["review_notes"],
        use_for_training=bool(row["use_for_training"]),
        linked_product_id=row["linked_product_id"],
        recognition=_json_loads(row["recognition_json"], {}),
        reviewed_at=row["reviewed_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class RecognitionRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _connect(self) -> sqlite3.Connection:
        return get_connection(self.settings)

    def create_from_response(
        self,
        *,
        image_bytes: bytes,
        content_type: str | None,
        source_name: str | None,
        response: ProductRecognitionResponse,
    ) -> RecognitionEventRecord:
        payload = response.model_dump()
        product = response.producto
        detection = response.deteccion
        ocr = response.ocr
        bbox = detection.bbox.model_dump()
        image_path = self._store_image_file(image_bytes, content_type)
        with self._connect() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO recognition_events (
                        trace_id, source_name, image_content_type, image_path,
                        predicted_nombre_producto, predicted_marca, predicted_tipo_producto,
                        predicted_presentacion, predicted_contenido_neto, predicted_unidad_medida,
                        predicted_categoria_sugerida, final_nombre_producto, final_marca,
                        final_tipo_producto, final_presentacion, final_contenido_neto,
                        final_unidad_medida, final_categoria_sugerida, yolo_confidence,
                        yolo_class_name, ocr_confidence, ocr_text, warnings_json,
                        bbox_json, recognition_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        response.trace_id,
                        source_name,
                        content_type,
                        str(image_path),
                        product.nombre_producto,
                        product.marca,
                        product.tipo_producto,
                        product.presentacion,
                        product.contenido_neto,
                        product.unidad_medida,
                        product.categoria_sugerida,
                        product.nombre_producto,
                        product.marca,
                        product.tipo_producto,
                        product.presentacion,
                        product.contenido_neto,
                        product.unidad_medida,
                        product.categoria_sugerida,
                        detection.confidence,
                        detection.class_name,
                        ocr.average_confidence,
                        ocr.text,
                        json.dumps(response.warnings, ensure_ascii=False),
                        json.dumps(bbox, ensure_ascii=False),
                        json.dumps(payload, ensure_ascii=False),
                    ),
                )
                conn.commit()
                event_id = cursor.lastrowid
            except Exception:
                image_path.unlink(missing_ok=True)
                raise
        return self.get(event_id)

    def _store_image_file(self, image_bytes: bytes, content_type: str | None) -> Path:
        image_dir = Path(self.settings.recognition_image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
        extension = self._extension_for_content_type(content_type)
        path = image_dir / f"{uuid4().hex}{extension}"
        path.write_bytes(image_bytes)
        return path

    def _extension_for_content_type(self, content_type: str | None) -> str:
        return {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }.get((content_type or "").lower(), ".img")

    def list(
        self,
        *,
        status: str | None = None,
        q: str | None = None,
        category: str | None = None,
        min_confidence: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RecognitionEventRecord]:
        where: list[str] = []
        params: list[Any] = []
        if status and status != "all":
            where.append("status = ?")
            params.append(status)
        if q:
            like = f"%{q.strip().lower()}%"
            where.append(
                """
                (
                    LOWER(COALESCE(final_nombre_producto, '')) LIKE ?
                    OR LOWER(COALESCE(predicted_nombre_producto, '')) LIKE ?
                    OR LOWER(COALESCE(final_marca, '')) LIKE ?
                    OR CAST(id AS TEXT) LIKE ?
                )
                """
            )
            params.extend([like, like, like, like])
        if category and category != "all":
            where.append("LOWER(COALESCE(final_categoria_sugerida, predicted_categoria_sugerida, '')) = ?")
            params.append(category.lower())
        if min_confidence is not None:
            where.append("COALESCE(yolo_confidence, 0) >= ?")
            params.append(min_confidence)

        sql = "SELECT * FROM recognition_events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_record(row) for row in rows]

    def get(self, event_id: int) -> RecognitionEventRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM recognition_events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            raise RecognitionEventNotFoundError(f"No existe reconocimiento con id {event_id}.")
        return _row_to_record(row)

    def get_image(self, event_id: int) -> tuple[bytes, str | None]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT image_blob, image_path, image_content_type FROM recognition_events WHERE id = ?",
                (event_id,),
            ).fetchone()
        if row is None:
            raise RecognitionEventNotFoundError(f"No existe imagen para reconocimiento {event_id}.")
        if row["image_path"]:
            path = Path(row["image_path"])
            if path.exists():
                return path.read_bytes(), row["image_content_type"]
        if row["image_blob"] is None:
            raise RecognitionEventNotFoundError(f"No existe imagen para reconocimiento {event_id}.")
        return bytes(row["image_blob"]), row["image_content_type"]

    def delete(self, event_id: int) -> None:
        image_path = self._image_path_for_delete(event_id)
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM recognition_events WHERE id = ?", (event_id,))
            conn.commit()
        if cursor.rowcount == 0:
            raise RecognitionEventNotFoundError(f"No existe reconocimiento con id {event_id}.")
        if image_path:
            image_path.unlink(missing_ok=True)

    def _image_path_for_delete(self, event_id: int) -> Path | None:
        with self._connect() as conn:
            row = conn.execute("SELECT image_path FROM recognition_events WHERE id = ?", (event_id,)).fetchone()
        if row is None or not row["image_path"]:
            return None
        path = Path(row["image_path"])
        image_dir = Path(self.settings.recognition_image_dir).resolve()
        try:
            resolved = path.resolve()
        except OSError:
            return None
        if image_dir == resolved.parent or image_dir in resolved.parents:
            return resolved
        return None

    def review(self, event_id: int, payload: dict[str, Any]) -> RecognitionEventRecord:
        status = payload.get("status")
        if status not in VALID_REVIEW_STATUSES:
            raise InvalidReviewStatusError(f"Estado inválido: {status}")

        fields = (
            "status",
            "final_nombre_producto",
            "final_marca",
            "final_tipo_producto",
            "final_presentacion",
            "final_contenido_neto",
            "final_unidad_medida",
            "final_categoria_sugerida",
            "final_codigo_barras",
            "failure_reason",
            "review_notes",
            "use_for_training",
            "linked_product_id",
        )
        assignments = ", ".join(f"{field} = ?" for field in fields)
        values = tuple(payload.get(field) for field in fields)
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE recognition_events
                SET {assignments}, reviewed_at = datetime('now'), updated_at = datetime('now')
                WHERE id = ?
                """,
                values + (event_id,),
            )
            conn.commit()
        if cursor.rowcount == 0:
            raise RecognitionEventNotFoundError(f"No existe reconocimiento con id {event_id}.")
        return self.get(event_id)

    def stats(self) -> dict[str, int | float]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS total FROM recognition_events GROUP BY status"
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) AS total FROM recognition_events").fetchone()["total"]
            training = conn.execute(
                "SELECT COUNT(*) AS total FROM recognition_events WHERE use_for_training = 1"
            ).fetchone()["total"]
        counts = {row["status"]: row["total"] for row in rows}
        validated = counts.get("validated", 0)
        corrected = counts.get("corrected", 0)
        rejected = counts.get("rejected", 0)
        reviewed = validated + corrected + rejected + counts.get("ignored", 0) + counts.get("duplicate", 0)
        precision = ((validated + corrected) / reviewed * 100) if reviewed else 0.0
        return {
            "pending_review": counts.get("pending_review", 0),
            "validated": validated,
            "corrected": corrected,
            "rejected": rejected,
            "training_candidates": training,
            "total": total,
            "precision": round(precision, 1),
        }
