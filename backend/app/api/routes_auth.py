import logging

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select

from app.api.deps import current_user_id, limiter_dep, settings_dep
from app.config import Settings
from app.db import get_factory
from app.models import User
from app.services import auth
from app.services.limits import Limiter

router = APIRouter(prefix="/auth", tags=["auth"])
log = logging.getLogger(__name__)
COOKIE = "rp_refresh"


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class TokenOut(BaseModel):
    access_token: str
    expires_in: int
    email: str
    user_id: str
    token_type: str = "bearer"


def _set_cookie(response: Response, token: str, settings: Settings) -> None:
    response.set_cookie(
        COOKIE, token, max_age=settings.refresh_token_days * 86400, httponly=True,
        secure=settings.cookie_secure, samesite="lax", path="/auth",
    )


def _check_origin(request: Request, settings: Settings) -> None:
    """The refresh cookie is sent automatically, so refuse requests from other sites."""
    origin = request.headers.get("origin")
    if origin and origin not in settings.cors_origins:
        raise HTTPException(403, "origin not allowed")


def _token_out(user: User, settings: Settings) -> TokenOut:
    token, ttl = auth.create_access_token(user.id, settings)
    return TokenOut(access_token=token, expires_in=ttl, email=user.email, user_id=str(user.id))


@router.post("/register", status_code=201)
async def register(
    body: Credentials, request: Request, response: Response,
    settings: Settings = Depends(settings_dep), limiter: Limiter = Depends(limiter_dep),
) -> TokenOut:
    _check_origin(request, settings)
    if not settings.registration_enabled:
        raise HTTPException(403, "registration is closed")
    ip = request.client.host if request.client else "unknown"
    ok, retry = await limiter.hit(f"reg:{ip}", 10, 3600)
    if not ok:
        raise HTTPException(429, "too many sign-ups from this address", headers={"Retry-After": str(retry)})
    if len(body.password) < auth.MIN_PASSWORD_LENGTH:
        raise HTTPException(422, f"password must be at least {auth.MIN_PASSWORD_LENGTH} characters")
    email = body.email.lower()
    with get_factory()() as session, session.begin():
        if session.scalar(select(User).where(User.email == email)) is not None:
            raise HTTPException(409, "that email is already registered")
        user = User(email=email, password_hash=auth.hash_password(body.password))
        session.add(user)
        session.flush()
        refresh = auth.issue_refresh_token(session, user.id, settings)
        out = _token_out(user, settings)
    _set_cookie(response, refresh, settings)
    return out


@router.post("/login")
async def login(
    body: Credentials, request: Request, response: Response,
    settings: Settings = Depends(settings_dep), limiter: Limiter = Depends(limiter_dep),
) -> TokenOut:
    _check_origin(request, settings)
    ip = request.client.host if request.client else "unknown"
    email = body.email.lower()
    ok, retry = await limiter.hit(f"login:{ip}:{email}", settings.login_attempts_per_15min, 900)
    if not ok:
        raise HTTPException(429, "too many attempts, try again later", headers={"Retry-After": str(retry)})
    with get_factory()() as session, session.begin():
        user = session.scalar(select(User).where(User.email == email))
        valid = auth.verify_password(body.password, user.password_hash if user else None)
        if user is None or not valid:
            # one message for both cases, and the same amount of hashing work
            raise HTTPException(401, "Invalid email or password")
        refresh = auth.issue_refresh_token(session, user.id, settings)
        out = _token_out(user, settings)
    _set_cookie(response, refresh, settings)
    return out


@router.post("/refresh")
def refresh(
    request: Request, response: Response, settings: Settings = Depends(settings_dep),
    rp_refresh: str | None = Cookie(default=None),
) -> TokenOut:
    _check_origin(request, settings)
    if not rp_refresh:
        raise HTTPException(401, "not signed in")
    session = get_factory()()
    try:
        try:
            user_id, new_token = auth.rotate_refresh_token(session, rp_refresh, settings)
            user = session.get(User, user_id)
            if user is None:
                raise auth.AuthError("user is gone")
            out = _token_out(user, settings)
            session.commit()
        except auth.AuthError:
            session.commit()  # keeps the revoke-all that reuse detection performs
            raise
    except auth.AuthError as exc:
        response.delete_cookie(COOKIE, path="/auth")
        raise HTTPException(401, "session expired, please sign in again") from exc
    finally:
        session.close()
    _set_cookie(response, new_token, settings)
    return out


@router.post("/logout", status_code=204)
def logout(
    request: Request, response: Response, settings: Settings = Depends(settings_dep),
    rp_refresh: str | None = Cookie(default=None),
) -> None:
    _check_origin(request, settings)
    if rp_refresh:
        with get_factory()() as session, session.begin():
            auth.revoke_token(session, rp_refresh)
    response.delete_cookie(COOKIE, path="/auth")


@router.get("/me")
def me(user_id=Depends(current_user_id)) -> dict[str, str]:
    with get_factory()() as session:
        user = session.get(User, user_id)
        if user is None:
            raise HTTPException(401, "unknown user")
        return {"user_id": str(user.id), "email": user.email}
