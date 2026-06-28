import re
import sqlite3
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

from app.core.config import Settings, get_settings

_lock = Lock()
_initialized: set[str] = set()


SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    password_salt TEXT NOT NULL,
    display_name TEXT,
    role TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

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
    reviewed_by_user_id INTEGER,
    reviewed_by_username TEXT,
    use_for_training INTEGER NOT NULL DEFAULT 0,
    linked_product_id INTEGER,
    recognition_json TEXT NOT NULL,
    reviewed_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(linked_product_id) REFERENCES productos(id),
    FOREIGN KEY(reviewed_by_user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_recognition_events_status ON recognition_events(status);
CREATE INDEX IF NOT EXISTS idx_recognition_events_created_at ON recognition_events(created_at);

CREATE TABLE IF NOT EXISTS inventory_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL,
    estado TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    closed_at TEXT
);

CREATE TABLE IF NOT EXISTS inventory_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    product_id INTEGER,
    recognition_event_id INTEGER,
    nombre_producto TEXT NOT NULL,
    marca TEXT,
    tipo_producto TEXT,
    categoria TEXT,
    contenido_neto TEXT,
    unidad_medida TEXT,
    cantidad INTEGER NOT NULL DEFAULT 1,
    ubicacion TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(session_id) REFERENCES inventory_sessions(id),
    FOREIGN KEY(product_id) REFERENCES productos(id),
    FOREIGN KEY(recognition_event_id) REFERENCES recognition_events(id)
);

CREATE INDEX IF NOT EXISTS idx_inventory_items_session ON inventory_items(session_id);
CREATE INDEX IF NOT EXISTS idx_inventory_items_category ON inventory_items(categoria);

CREATE TABLE IF NOT EXISTS product_stock_counts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mobile_product_id TEXT,
    nombre_producto TEXT NOT NULL,
    cantidad_final INTEGER NOT NULL,
    confianza REAL NOT NULL DEFAULT 0,
    total_fotos INTEGER NOT NULL DEFAULT 0,
    valid_fotos INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'mobile',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS product_stock_count_photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    count_id INTEGER NOT NULL,
    recognition_event_id INTEGER,
    source_name TEXT,
    detected_name TEXT,
    matched INTEGER NOT NULL DEFAULT 0,
    accepted INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0,
    warnings_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(count_id) REFERENCES product_stock_counts(id),
    FOREIGN KEY(recognition_event_id) REFERENCES recognition_events(id)
);

CREATE INDEX IF NOT EXISTS idx_product_stock_counts_mobile_product ON product_stock_counts(mobile_product_id);
CREATE INDEX IF NOT EXISTS idx_product_stock_count_photos_count ON product_stock_count_photos(count_id);
"""


MYSQL_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS productos (
        id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        nombre_producto VARCHAR(255) NOT NULL,
        marca VARCHAR(120),
        tipo_producto VARCHAR(120),
        presentacion VARCHAR(120),
        contenido_neto VARCHAR(60),
        unidad_medida VARCHAR(20),
        categoria_sugerida VARCHAR(120),
        codigo_barras VARCHAR(64) UNIQUE,
        precio_venta DECIMAL(10, 2) NOT NULL DEFAULT 0,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_productos_nombre (nombre_producto)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS recognition_events (
        id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        trace_id VARCHAR(120) NOT NULL,
        source_name VARCHAR(255),
        image_content_type VARCHAR(120),
        image_blob LONGBLOB,
        image_path VARCHAR(500),
        status VARCHAR(40) NOT NULL DEFAULT 'pending_review',
        predicted_nombre_producto VARCHAR(255),
        predicted_marca VARCHAR(120),
        predicted_tipo_producto VARCHAR(120),
        predicted_presentacion VARCHAR(120),
        predicted_contenido_neto VARCHAR(60),
        predicted_unidad_medida VARCHAR(20),
        predicted_categoria_sugerida VARCHAR(120),
        final_nombre_producto VARCHAR(255),
        final_marca VARCHAR(120),
        final_tipo_producto VARCHAR(120),
        final_presentacion VARCHAR(120),
        final_contenido_neto VARCHAR(60),
        final_unidad_medida VARCHAR(20),
        final_categoria_sugerida VARCHAR(120),
        final_codigo_barras VARCHAR(64),
        yolo_confidence DOUBLE,
        yolo_class_name VARCHAR(120),
        ocr_confidence DOUBLE,
        ocr_text TEXT,
        warnings_json JSON NOT NULL DEFAULT (JSON_ARRAY()),
        bbox_json JSON,
        failure_reason VARCHAR(120),
        review_notes VARCHAR(500),
        reviewed_by_user_id INT,
        reviewed_by_username VARCHAR(120),
        use_for_training TINYINT(1) NOT NULL DEFAULT 0,
        linked_product_id INT,
        recognition_json JSON NOT NULL,
        reviewed_at DATETIME,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_recognition_events_status (status),
        INDEX idx_recognition_events_created_at (created_at),
        CONSTRAINT fk_recognition_product FOREIGN KEY (linked_product_id) REFERENCES productos(id)
            ON DELETE SET NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS inventory_sessions (
        id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        nombre VARCHAR(120) NOT NULL,
        estado VARCHAR(40) NOT NULL DEFAULT 'open',
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        closed_at DATETIME
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS inventory_items (
        id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        session_id INT NOT NULL,
        product_id INT,
        recognition_event_id INT,
        nombre_producto VARCHAR(255) NOT NULL,
        marca VARCHAR(120),
        tipo_producto VARCHAR(120),
        categoria VARCHAR(120),
        contenido_neto VARCHAR(60),
        unidad_medida VARCHAR(20),
        cantidad INT NOT NULL DEFAULT 1,
        ubicacion VARCHAR(120),
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_inventory_items_session (session_id),
        INDEX idx_inventory_items_category (categoria),
        CONSTRAINT fk_inventory_session FOREIGN KEY (session_id) REFERENCES inventory_sessions(id)
            ON DELETE CASCADE,
        CONSTRAINT fk_inventory_product FOREIGN KEY (product_id) REFERENCES productos(id)
            ON DELETE SET NULL,
        CONSTRAINT fk_inventory_recognition FOREIGN KEY (recognition_event_id) REFERENCES recognition_events(id)
            ON DELETE SET NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS product_stock_counts (
        id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        mobile_product_id VARCHAR(120),
        nombre_producto VARCHAR(255) NOT NULL,
        cantidad_final INT NOT NULL,
        confianza DOUBLE NOT NULL DEFAULT 0,
        total_fotos INT NOT NULL DEFAULT 0,
        valid_fotos INT NOT NULL DEFAULT 0,
        source VARCHAR(40) NOT NULL DEFAULT 'mobile',
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_product_stock_counts_mobile_product (mobile_product_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS product_stock_count_photos (
        id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        count_id INT NOT NULL,
        recognition_event_id INT,
        source_name VARCHAR(255),
        detected_name VARCHAR(255),
        matched TINYINT(1) NOT NULL DEFAULT 0,
        accepted TINYINT(1) NOT NULL DEFAULT 0,
        confidence DOUBLE NOT NULL DEFAULT 0,
        warnings_json JSON NOT NULL DEFAULT (JSON_ARRAY()),
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_product_stock_count_photos_count (count_id),
        CONSTRAINT fk_stock_photo_count FOREIGN KEY (count_id) REFERENCES product_stock_counts(id)
            ON DELETE CASCADE,
        CONSTRAINT fk_stock_photo_recognition FOREIGN KEY (recognition_event_id) REFERENCES recognition_events(id)
            ON DELETE SET NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
]


class MySqlCursorResult:
    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor
        self.lastrowid = cursor.lastrowid
        self.rowcount = cursor.rowcount

    def fetchone(self):
        return _normalize_mysql_row(self._cursor.fetchone())

    def fetchall(self):
        return [_normalize_mysql_row(row) for row in self._cursor.fetchall()]


class MySqlConnection:
    def __init__(self, raw_conn: Any) -> None:
        self._conn = raw_conn

    def __enter__(self) -> "MySqlConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type:
            self._conn.rollback()
        self.close()

    def execute(self, sql: str, params: Iterable[Any] | None = None) -> MySqlCursorResult:
        cursor = self._conn.cursor()
        cursor.execute(_to_mysql_sql(sql), tuple(params or ()))
        return MySqlCursorResult(cursor)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()


def _to_mysql_sql(sql: str) -> str:
    sql = sql.replace("datetime('now')", "CURRENT_TIMESTAMP")
    sql = sql.replace("CAST(id AS TEXT)", "CAST(id AS CHAR)")
    sql = re.sub(r"\bLENGTH\(", "CHAR_LENGTH(", sql)
    return sql.replace("?", "%s")


def _normalize_mysql_row(row):
    if row is None:
        return None
    return {key: _normalize_mysql_value(value) for key, value in row.items()}


def _normalize_mysql_value(value):
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _resolve_path(settings: Settings) -> Path:
    path = Path(settings.sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def init_db(settings: Settings | None = None):
    settings = settings or get_settings()
    if settings.db_backend.lower() == "mysql":
        return init_mysql_db(settings)
    return init_sqlite_db(settings)


def init_sqlite_db(settings: Settings) -> Path:
    path = _resolve_path(settings)
    key = f"sqlite:{path.resolve()}"
    with _lock:
        if key in _initialized:
            return path
        with sqlite3.connect(path) as conn:
            conn.executescript(SQLITE_SCHEMA)
            _apply_sqlite_lightweight_migrations(conn)
            conn.commit()
        _initialized.add(key)
    return path


def init_mysql_db(settings: Settings) -> str:
    import pymysql

    key = f"mysql:{settings.mysql_host}:{settings.mysql_port}:{settings.mysql_database}"
    with _lock:
        if key in _initialized:
            return settings.mysql_database
        root_conn = pymysql.connect(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            charset=settings.mysql_charset,
            autocommit=True,
        )
        try:
            with root_conn.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{settings.mysql_database}` "
                    f"CHARACTER SET {settings.mysql_charset} COLLATE {settings.mysql_charset}_unicode_ci"
                )
        finally:
            root_conn.close()

        conn = _mysql_raw_connection(settings)
        try:
            with conn.cursor() as cursor:
                for statement in MYSQL_SCHEMA:
                    cursor.execute(statement)
            conn.commit()
        finally:
            conn.close()
        _initialized.add(key)
    return settings.mysql_database


def _mysql_raw_connection(settings: Settings):
    import pymysql

    return pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
        charset=settings.mysql_charset,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def _apply_sqlite_lightweight_migrations(conn: sqlite3.Connection) -> None:
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(recognition_events)").fetchall()
    }
    if "image_path" not in columns:
        conn.execute("ALTER TABLE recognition_events ADD COLUMN image_path TEXT")
    if "reviewed_by_user_id" not in columns:
        conn.execute("ALTER TABLE recognition_events ADD COLUMN reviewed_by_user_id INTEGER")
    if "reviewed_by_username" not in columns:
        conn.execute("ALTER TABLE recognition_events ADD COLUMN reviewed_by_username TEXT")


def get_connection(settings: Settings | None = None):
    settings = settings or get_settings()
    if settings.db_backend.lower() == "mysql":
        init_mysql_db(settings)
        return MySqlConnection(_mysql_raw_connection(settings))

    path = init_sqlite_db(settings)
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
