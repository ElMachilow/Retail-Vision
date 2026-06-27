import sqlite3
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.core.security import hash_password, verify_password
from app.db.connection import get_connection


@dataclass(frozen=True)
class UserRecord:
    id: int
    username: str
    display_name: str | None
    role: str
    created_at: str
    updated_at: str


def _row_to_user(row: sqlite3.Row) -> UserRecord:
    return UserRecord(
        id=row["id"],
        username=row["username"],
        display_name=row["display_name"],
        role=row["role"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class UserRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _connect(self) -> sqlite3.Connection:
        return get_connection(self.settings)

    def create_user(
        self,
        username: str,
        password: str,
        display_name: str | None = None,
        role: str = "user",
    ) -> UserRecord:
        password_hash, password_salt = hash_password(password)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (
                    username, password_hash, password_salt, display_name, role
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (username, password_hash, password_salt, display_name, role),
            )
            conn.commit()
            return self.get_by_id(cursor.lastrowid)

    def get_by_username(self, username: str) -> UserRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        return _row_to_user(row) if row else None

    def get_by_id(self, user_id: int) -> UserRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return _row_to_user(row) if row else None

    def authenticate(self, username: str, password: str) -> UserRecord | None:
        user = self.get_by_username(username)
        if user is None:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT password_hash, password_salt FROM users WHERE id = ?",
                (user.id,),
            ).fetchone()
        if not row:
            return None
        if verify_password(password, row["password_hash"], row["password_salt"]):
            return user
        return None

    def ensure_default_admin_user(self, username: str, password: str) -> UserRecord | None:
        existing = self.get_by_username(username)
        if existing:
            return existing
        return self.create_user(username, password, display_name="Administrador", role="admin")
