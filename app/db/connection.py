import sqlite3
from pathlib import Path
from threading import Lock

from app.core.config import Settings, get_settings

_lock = Lock()
_initialized: set[str] = set()


SCHEMA = """
CREATE TABLE IF NOT EXISTS productos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre_producto TEXT NOT NULL,
    marca TEXT,
    tipo_producto TEXT,
    presentacion TEXT,
    contenido_neto TEXT,
    unidad_medida TEXT,
    categoria_sugerida TEXT,
    codigo_barras TEXT UNIQUE,
    precio_venta REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_productos_nombre ON productos(nombre_producto);

CREATE TABLE IF NOT EXISTS recognition_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    source_name TEXT,
    image_content_type TEXT,
    image_blob BLOB,
    image_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending_review',
    predicted_nombre_producto TEXT,
    predicted_marca TEXT,
    predicted_tipo_producto TEXT,
    predicted_presentacion TEXT,
    predicted_contenido_neto TEXT,
    predicted_unidad_medida TEXT,
    predicted_categoria_sugerida TEXT,
    final_nombre_producto TEXT,
    final_marca TEXT,
    final_tipo_producto TEXT,
    final_presentacion TEXT,
    final_contenido_neto TEXT,
    final_unidad_medida TEXT,
    final_categoria_sugerida TEXT,
    final_codigo_barras TEXT,
    yolo_confidence REAL,
    yolo_class_name TEXT,
    ocr_confidence REAL,
    ocr_text TEXT,
    warnings_json TEXT NOT NULL DEFAULT '[]',
    bbox_json TEXT,
    failure_reason TEXT,
    review_notes TEXT,
    use_for_training INTEGER NOT NULL DEFAULT 0,
    linked_product_id INTEGER,
    recognition_json TEXT NOT NULL,
    reviewed_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(linked_product_id) REFERENCES productos(id)
);

CREATE INDEX IF NOT EXISTS idx_recognition_events_status ON recognition_events(status);
CREATE INDEX IF NOT EXISTS idx_recognition_events_created_at ON recognition_events(created_at);
"""


def _resolve_path(settings: Settings) -> Path:
    path = Path(settings.sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def init_db(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    path = _resolve_path(settings)
    key = str(path.resolve())
    with _lock:
        if key in _initialized:
            return path
        with sqlite3.connect(path) as conn:
            conn.executescript(SCHEMA)
            _apply_lightweight_migrations(conn)
            conn.commit()
        _initialized.add(key)
    return path


def _apply_lightweight_migrations(conn: sqlite3.Connection) -> None:
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(recognition_events)").fetchall()
    }
    if "image_path" not in columns:
        conn.execute("ALTER TABLE recognition_events ADD COLUMN image_path TEXT")


def get_connection(settings: Settings | None = None) -> sqlite3.Connection:
    settings = settings or get_settings()
    path = init_db(settings)
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
