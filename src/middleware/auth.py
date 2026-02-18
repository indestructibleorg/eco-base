"""
Authentication middleware - JWT + API Key verification.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import jwt
from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from src.config.settings import get_settings
from src.schemas.auth import APIKeyCreate, APIKeyInfo, APIKeyResponse, UserRole
from src.utils.logging import get_logger

logger = get_logger("superai.auth")

_bearer_scheme = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthMiddleware:
    """Manages API keys and JWT tokens."""

    def __init__(self):
        self._settings = get_settings()
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        self._init_default_key()

    def _init_default_key(self) -> None:
        """Create a default admin API key for initial setup."""
        default_key = f"sk-superai-{secrets.token_hex(24)}"
        key_hash = self._hash_key(default_key)
        self._api_keys[key_hash] = {
            "key_id": "default-admin",
            "name": "Default Admin Key",
            "prefix": default_key[:12],
            "role": UserRole.ADMIN,
            "rate_limit_per_minute": 10000,
            "allowed_models": [],
            "created_at": datetime.now(timezone.utc),
            "expires_at": None,
            "last_used_at": None,
            "total_requests": 0,
            "is_active": True,
        }
        logger.info(
            "Default admin API key created",
            prefix=default_key[:12],
            full_key_logged_once=default_key,
        )

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def create_api_key(self, request: APIKeyCreate) -> APIKeyResponse:
        """Generate a new API key."""
        raw_key = f"sk-superai-{secrets.token_hex(24)}"
        key_hash = self._hash_key(raw_key)
        now = datetime.now(timezone.utc)

        info = APIKeyInfo(
            key_id=secrets.token_hex(8),
            name=request.name,
            prefix=raw_key[:12],
            role=request.role,
            rate_limit_per_minute=request.rate_limit_per_minute,
            allowed_models=request.allowed_models,
            created_at=now,
            expires_at=request.expires_at,
            last_used_at=None,
            total_requests=0,
            is_active=True,
        )

        self._api_keys[key_hash] = info.model_dump()
        logger.info("API key created", key_id=info.key_id, name=info.name, role=info.role.value)

        return APIKeyResponse(key=raw_key, info=info)

    def validate_api_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key and return its metadata."""
        key_hash = self._hash_key(key)
        key_data = self._api_keys.get(key_hash)

        if not key_data:
            return None

        if not key_data.get("is_active", True):
            return None

        expires = key_data.get("expires_at")
        if expires and isinstance(expires, datetime) and expires < datetime.now(timezone.utc):
            return None

        key_data["last_used_at"] = datetime.now(timezone.utc)
        key_data["total_requests"] = key_data.get("total_requests", 0) + 1

        return key_data

    def create_jwt_token(self, user_id: str, role: str) -> str:
        """Create a JWT access token."""
        payload = {
            "sub": user_id,
            "role": role,
            "iat": int(time.time()),
            "exp": int(time.time()) + self._settings.jwt_expiration_minutes * 60,
        }
        return jwt.encode(payload, self._settings.secret_key, algorithm=self._settings.jwt_algorithm)

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(
                token,
                self._settings.secret_key,
                algorithms=[self._settings.jwt_algorithm],
            )
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key by key_id."""
        for key_hash, data in self._api_keys.items():
            if data.get("key_id") == key_id:
                data["is_active"] = False
                logger.info("API key revoked", key_id=key_id)
                return True
        return False

    def list_api_keys(self) -> List[APIKeyInfo]:
        """List all API keys (without secrets)."""
        results = []
        for data in self._api_keys.values():
            results.append(APIKeyInfo(**{
                k: v for k, v in data.items()
                if k in APIKeyInfo.model_fields
            }))
        return results


# Singleton
_auth: Optional[AuthMiddleware] = None


def get_auth() -> AuthMiddleware:
    global _auth
    if _auth is None:
        _auth = AuthMiddleware()
    return _auth


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Security(_api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
) -> Dict[str, Any]:
    """FastAPI dependency for API key / JWT verification."""
    auth = get_auth()

    # Try API key header first
    if api_key:
        key_data = auth.validate_api_key(api_key)
        if key_data:
            return key_data

    # Try Bearer token
    if bearer and bearer.credentials:
        # Check if it's an API key format
        if bearer.credentials.startswith("sk-"):
            key_data = auth.validate_api_key(bearer.credentials)
            if key_data:
                return key_data
        # Try JWT
        jwt_data = auth.verify_jwt_token(bearer.credentials)
        if jwt_data:
            return {"role": jwt_data.get("role", "viewer"), "user_id": jwt_data.get("sub")}

    raise HTTPException(
        status_code=401,
        detail="Invalid or missing authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )