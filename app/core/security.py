import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

from fastapi import HTTPException, Request, status

from app.core.config import Settings, get_settings

SESSION_COOKIE = "visionai_user_session"


def hash_password(password: str) -> tuple[str, str]:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    )
    return digest.hex(), salt


def verify_password(password: str, password_hash: str, password_salt: str) -> bool:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        password_salt.encode("utf-8"),
        100_000,
    )
    return secrets.compare_digest(digest.hex(), password_hash)


def create_session_cookie(user_id: int, username: str, role: str, settings: Settings) -> str:
    payload = {
        "sub": username,
        "role": role,
        "user_id": user_id,
        "exp": int(time.time()) + settings.admin_session_max_age_seconds,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded_payload = _urlsafe_b64encode(payload_bytes)
    signature = _sign(encoded_payload, settings.admin_session_secret)
    return f"{encoded_payload}.{signature}"


def decode_session_cookie(value: str, secret: str) -> dict[str, Any] | None:
    try:
        encoded_payload, signature = value.split(".", 1)
    except ValueError:
        return None
    if not secrets.compare_digest(signature, _sign(encoded_payload, secret)):
        return None
    try:
        payload = json.loads(_urlsafe_b64decode(encoded_payload))
    except json.JSONDecodeError:
        return None
    return payload


def get_current_user(request: Request, settings: Settings | None = None) -> dict[str, Any] | None:
    settings = settings or get_settings()
    raw_cookie = request.cookies.get(SESSION_COOKIE)
    if not raw_cookie:
        return None
    payload = decode_session_cookie(raw_cookie, settings.admin_session_secret)
    if not payload:
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    return payload


def is_admin_request(request: Request, settings: Settings | None = None) -> bool:
    payload = get_current_user(request, settings)
    if not payload:
        return False
    return payload.get("role") == "admin"


def require_admin(request: Request) -> None:
    if not is_admin_request(request):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Autenticacion requerida para recursos administrativos.",
        )


def _sign(encoded_payload: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), encoded_payload.encode("utf-8"), hashlib.sha256).digest()
    return _urlsafe_b64encode(digest)


def _urlsafe_b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _urlsafe_b64decode(value: str) -> Any:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding).decode("utf-8")
