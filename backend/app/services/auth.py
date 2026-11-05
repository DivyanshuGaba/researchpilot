"""Passwords (argon2id), short-lived JWT access tokens, rotating refresh tokens."""

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import RefreshToken

_hasher = PasswordHasher()
# Verified against when the email is unknown, so "no such user" costs the same as "wrong password".
_DUMMY_HASH = _hasher.hash("dummy-password-for-timing")
MIN_PASSWORD_LENGTH = 10
INSECURE_SECRETS = {"", "change_me", "changeme", "secret"}


class AuthError(Exception):
    pass


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and bool(password_hash)
    except (VerificationError, InvalidHashError):
        return False


def create_access_token(user_id: uuid.UUID, settings: Settings) -> tuple[str, int]:
    now = datetime.now(UTC)
    exp = now + timedelta(minutes=settings.access_token_minutes)
    token = jwt.encode(
        {"sub": str(user_id), "iat": now, "exp": exp, "typ": "access"},
        settings.jwt_secret,
        algorithm="HS256",
    )
    return token, int((exp - now).total_seconds())


def decode_access_token(token: str, settings: Settings) -> uuid.UUID:
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],  # never accept the algorithm named in the token header
            options={"require": ["exp", "sub", "typ"]},
        )
        if claims.get("typ") != "access":
            raise AuthError("wrong token type")
        return uuid.UUID(claims["sub"])
    except (jwt.PyJWTError, ValueError) as exc:
        raise AuthError("invalid or expired token") from exc


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def issue_refresh_token(session: Session, user_id: uuid.UUID, settings: Settings) -> str:
    token = secrets.token_urlsafe(32)
    session.add(
        RefreshToken(
            user_id=user_id,
            token_hash=_digest(token),
            expires_at=datetime.now(UTC) + timedelta(days=settings.refresh_token_days),
        )
    )
    session.flush()
    return token


def rotate_refresh_token(session: Session, token: str, settings: Settings) -> tuple[uuid.UUID, str]:
    """Exchange a refresh token for a new one. Presenting an already-used token means it was
    probably stolen, so every session of that user is revoked."""
    row = session.scalar(select(RefreshToken).where(RefreshToken.token_hash == _digest(token)))
    now = datetime.now(UTC)
    if row is None:
        raise AuthError("unknown refresh token")
    if row.revoked_at is not None:
        revoke_all(session, row.user_id)
        raise AuthError("refresh token was already used")
    if row.expires_at < now:
        raise AuthError("refresh token expired")
    row.revoked_at = now
    return row.user_id, issue_refresh_token(session, row.user_id, settings)


def revoke_token(session: Session, token: str) -> None:
    session.execute(
        update(RefreshToken)
        .where(RefreshToken.token_hash == _digest(token), RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(UTC))
    )


def revoke_all(session: Session, user_id: uuid.UUID) -> None:
    session.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(UTC))
    )


def assert_safe_config(settings: Settings) -> None:
    if settings.environment != "dev" and settings.jwt_secret in INSECURE_SECRETS:
        raise RuntimeError("JWT_SECRET must be set to a long random value outside dev")
    if len(settings.jwt_secret.encode()) < 16 and settings.environment != "dev":
        raise RuntimeError("JWT_SECRET is too short")
