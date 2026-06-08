import sqlite3
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.core.exceptions import AppError
from app.db.connection import get_connection


class InventorySessionNotFoundError(AppError):
    status_code = 404
    error_code = "INVENTORY_SESSION_NOT_FOUND"


class InventoryItemNotFoundError(AppError):
    status_code = 404
    error_code = "INVENTORY_ITEM_NOT_FOUND"


@dataclass(frozen=True)
class InventorySessionRecord:
    id: int
    nombre: str
    estado: str
    created_at: str
    closed_at: str | None


@dataclass(frozen=True)
class InventoryItemRecord:
    id: int
    session_id: int
    product_id: int | None
    recognition_event_id: int | None
    nombre_producto: str
    marca: str | None
    tipo_producto: str | None
    categoria: str | None
    contenido_neto: str | None
    unidad_medida: str | None
    cantidad: int
    ubicacion: str | None
    created_at: str
    updated_at: str


def _session_from_row(row: sqlite3.Row) -> InventorySessionRecord:
    return InventorySessionRecord(
        id=row["id"],
        nombre=row["nombre"],
        estado=row["estado"],
        created_at=row["created_at"],
        closed_at=row["closed_at"],
    )


def _item_from_row(row: sqlite3.Row) -> InventoryItemRecord:
    return InventoryItemRecord(
        id=row["id"],
        session_id=row["session_id"],
        product_id=row["product_id"],
        recognition_event_id=row["recognition_event_id"],
        nombre_producto=row["nombre_producto"],
        marca=row["marca"],
        tipo_producto=row["tipo_producto"],
        categoria=row["categoria"],
        contenido_neto=row["contenido_neto"],
        unidad_medida=row["unidad_medida"],
        cantidad=int(row["cantidad"]),
        ubicacion=row["ubicacion"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class InventoryRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _connect(self) -> sqlite3.Connection:
        return get_connection(self.settings)

    def create_session(self, nombre: str) -> InventorySessionRecord:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO inventory_sessions (nombre) VALUES (?)",
                (nombre,),
            )
            conn.commit()
            session_id = cursor.lastrowid
        return self.get_session(session_id)

    def list_sessions(self, limit: int = 100, offset: int = 0) -> list[InventorySessionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM inventory_sessions
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [_session_from_row(row) for row in rows]

    def get_session(self, session_id: int) -> InventorySessionRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM inventory_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise InventorySessionNotFoundError(f"No existe sesion de inventario con id {session_id}.")
        return _session_from_row(row)

    def close_session(self, session_id: int) -> InventorySessionRecord:
        self.get_session(session_id)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE inventory_sessions
                SET estado = 'closed', closed_at = datetime('now')
                WHERE id = ?
                """,
                (session_id,),
            )
            conn.commit()
        return self.get_session(session_id)

    def create_item(self, session_id: int, payload: dict[str, Any]) -> InventoryItemRecord:
        self.get_session(session_id)
        fields = (
            "session_id",
            "product_id",
            "recognition_event_id",
            "nombre_producto",
            "marca",
            "tipo_producto",
            "categoria",
            "contenido_neto",
            "unidad_medida",
            "cantidad",
            "ubicacion",
        )
        values = (
            session_id,
            payload.get("product_id"),
            payload.get("recognition_event_id"),
            payload.get("nombre_producto"),
            payload.get("marca"),
            payload.get("tipo_producto"),
            payload.get("categoria"),
            payload.get("contenido_neto"),
            payload.get("unidad_medida"),
            payload.get("cantidad"),
            payload.get("ubicacion"),
        )
        with self._connect() as conn:
            cursor = conn.execute(
                f"INSERT INTO inventory_items ({', '.join(fields)}) VALUES ({', '.join('?' for _ in fields)})",
                values,
            )
            conn.commit()
            item_id = cursor.lastrowid
        return self.get_item(item_id)

    def get_item(self, item_id: int) -> InventoryItemRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM inventory_items WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            raise InventoryItemNotFoundError(f"No existe item de inventario con id {item_id}.")
        return _item_from_row(row)

    def list_items(self, session_id: int, limit: int = 200, offset: int = 0) -> list[InventoryItemRecord]:
        self.get_session(session_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM inventory_items
                WHERE session_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (session_id, limit, offset),
            ).fetchall()
        return [_item_from_row(row) for row in rows]

    def summary(self, session_id: int) -> dict[str, Any]:
        self.get_session(session_id)
        with self._connect() as conn:
            category_rows = conn.execute(
                """
                SELECT COALESCE(categoria, 'Sin categoria') AS categoria,
                       COUNT(*) AS productos,
                       COALESCE(SUM(cantidad), 0) AS unidades
                FROM inventory_items
                WHERE session_id = ?
                GROUP BY COALESCE(categoria, 'Sin categoria')
                ORDER BY unidades DESC, categoria ASC
                """,
                (session_id,),
            ).fetchall()
            totals = conn.execute(
                """
                SELECT COUNT(*) AS productos, COALESCE(SUM(cantidad), 0) AS unidades
                FROM inventory_items
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        return {
            "session_id": session_id,
            "total_productos": int(totals["productos"] or 0),
            "total_unidades": int(totals["unidades"] or 0),
            "categorias": [
                {
                    "categoria": row["categoria"],
                    "productos": int(row["productos"] or 0),
                    "unidades": int(row["unidades"] or 0),
                }
                for row in category_rows
            ],
        }
