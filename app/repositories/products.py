import sqlite3
from dataclasses import dataclass

from app.core.config import Settings
from app.core.exceptions import AppError
from app.db.connection import get_connection


class DuplicateBarcodeError(AppError):
    status_code = 409
    error_code = "DUPLICATE_BARCODE"


class ProductNotFoundError(AppError):
    status_code = 404
    error_code = "PRODUCT_NOT_FOUND"


@dataclass(frozen=True)
class ProductRecord:
    id: int
    nombre_producto: str
    marca: str | None
    tipo_producto: str | None
    presentacion: str | None
    contenido_neto: str | None
    unidad_medida: str | None
    categoria_sugerida: str | None
    codigo_barras: str | None
    precio_venta: float
    created_at: str
    updated_at: str


def _row_to_record(row: sqlite3.Row) -> ProductRecord:
    return ProductRecord(
        id=row["id"],
        nombre_producto=row["nombre_producto"],
        marca=row["marca"],
        tipo_producto=row["tipo_producto"],
        presentacion=row["presentacion"],
        contenido_neto=row["contenido_neto"],
        unidad_medida=row["unidad_medida"],
        categoria_sugerida=row["categoria_sugerida"],
        codigo_barras=row["codigo_barras"],
        precio_venta=float(row["precio_venta"] or 0),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class ProductRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _connect(self) -> sqlite3.Connection:
        return get_connection(self.settings)

    def list_all(self, limit: int = 100, offset: int = 0) -> list[ProductRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM productos ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def get(self, product_id: int) -> ProductRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM productos WHERE id = ?", (product_id,)).fetchone()
        if row is None:
            raise ProductNotFoundError(f"No existe producto con id {product_id}.")
        return _row_to_record(row)

    def search_by_name(self, query: str, limit: int = 3) -> list[ProductRecord]:
        like = f"%{query.strip().lower()}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM productos
                WHERE LOWER(nombre_producto) LIKE ?
                ORDER BY
                    CASE WHEN LOWER(nombre_producto) LIKE ? THEN 0 ELSE 1 END,
                    LENGTH(nombre_producto) ASC,
                    updated_at DESC
                LIMIT ?
                """,
                (like, f"{query.strip().lower()}%", limit),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def create(self, payload: dict) -> ProductRecord:
        fields = (
            "nombre_producto",
            "marca",
            "tipo_producto",
            "presentacion",
            "contenido_neto",
            "unidad_medida",
            "categoria_sugerida",
            "codigo_barras",
            "precio_venta",
        )
        values = tuple(payload.get(field) for field in fields)
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    f"INSERT INTO productos ({', '.join(fields)}) VALUES ({', '.join('?' for _ in fields)})",
                    values,
                )
                conn.commit()
                product_id = cursor.lastrowid
        except Exception as exc:
            if "codigo_barras" in str(exc):
                raise DuplicateBarcodeError("El código de barras ya está registrado.") from exc
            raise
        return self.get(product_id)

    def update(self, product_id: int, payload: dict) -> ProductRecord:
        existing = self.get(product_id)
        fields = (
            "nombre_producto",
            "marca",
            "tipo_producto",
            "presentacion",
            "contenido_neto",
            "unidad_medida",
            "categoria_sugerida",
            "codigo_barras",
            "precio_venta",
        )
        merged = {field: payload.get(field, getattr(existing, field)) for field in fields}
        assignments = ", ".join(f"{field} = ?" for field in fields)
        values = tuple(merged[field] for field in fields) + (product_id,)
        try:
            with self._connect() as conn:
                conn.execute(
                    f"UPDATE productos SET {assignments}, updated_at = datetime('now') WHERE id = ?",
                    values,
                )
                conn.commit()
        except Exception as exc:
            if "codigo_barras" in str(exc):
                raise DuplicateBarcodeError("El código de barras ya está registrado.") from exc
            raise
        return self.get(product_id)
