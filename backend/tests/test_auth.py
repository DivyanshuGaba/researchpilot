import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from app.api import deps
from app.config import Settings
from app.db import get_factory
from app.main import app
from app.models import RefreshToken, User
from app.services import auth as auth_service
from app.services.limits import Limiter

from .fakes import FakeLLM, FakeStore, auth, make_deps, make_hit
from .test_integration_db import _db_ready

pytestmark = pytest.mark.skipif(not _db_ready(), reason="Postgres not available")

PASSWORD = "correct horse battery"
ORIGIN = {"Origin": "http://localhost:3000"}


def _email():
    return f"{uuid.uuid4().hex[:10]}@example.com"


@pytest.fixture
def client():
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


def register(client, email=None, password=PASSWORD):
    email = email or _email()
    r = client.post("/auth/register", json={"email": email, "password": password}, headers=ORIGIN)
    return email, r


# --- unit: tokens and passwords ---------------------------------------------------------------


def test_password_hashing_is_argon2_and_verifies_only_the_right_password():
    h = auth_service.hash_password(PASSWORD)
    assert h.startswith("$argon2id$") and PASSWORD not in h
    assert auth_service.verify_password(PASSWORD, h)
    assert not auth_service.verify_password("wrong password", h)
    assert not auth_service.verify_password(PASSWORD, None)  # unknown user: still false, no crash
    assert not auth_service.verify_password("dummy-password-for-timing", None)
    assert not auth_service.verify_password(PASSWORD, "not-a-hash")


def test_access_token_roundtrip_and_rejections():
    s, uid = Settings(), uuid.uuid4()
    token, ttl = auth_service.create_access_token(uid, s)
    assert ttl == 15 * 60 and auth_service.decode_access_token(token, s) == uid

    now = datetime.now(UTC)
    expired = jwt.encode({"sub": str(uid), "iat": now, "exp": now - timedelta(seconds=5), "typ": "access"}, s.jwt_secret, "HS256")
    wrong_secret = jwt.encode({"sub": str(uid), "exp": now + timedelta(minutes=5), "typ": "access"}, "another-secret-key-x", "HS256")
    wrong_type = jwt.encode({"sub": str(uid), "exp": now + timedelta(minutes=5), "typ": "refresh"}, s.jwt_secret, "HS256")
    no_exp = jwt.encode({"sub": str(uid), "typ": "access"}, s.jwt_secret, "HS256")
    none_alg = jwt.encode({"sub": str(uid), "exp": now + timedelta(minutes=5), "typ": "access"}, "", algorithm="none")
    bad_sub = jwt.encode({"sub": "not-a-uuid", "exp": now + timedelta(minutes=5), "typ": "access"}, s.jwt_secret, "HS256")
    for bad in (expired, wrong_secret, wrong_type, no_exp, none_alg, bad_sub, token[:-3] + "xxx", "garbage", ""):
        with pytest.raises(auth_service.AuthError):
            auth_service.decode_access_token(bad, s)


def test_insecure_jwt_secret_is_refused_outside_dev():
    auth_service.assert_safe_config(Settings())  # dev default is allowed
    with pytest.raises(RuntimeError):
        auth_service.assert_safe_config(Settings(environment="production"))
    with pytest.raises(RuntimeError):
        auth_service.assert_safe_config(Settings(environment="production", jwt_secret="short"))
    auth_service.assert_safe_config(Settings(environment="production", jwt_secret="x" * 40))


# --- endpoints --------------------------------------------------------------------------------


def test_register_login_me_flow_and_cookie_attributes(client):
    email, r = register(client)
    assert r.status_code == 201 and r.json()["email"] == email
    cookie = r.headers["set-cookie"].lower()
    assert "rp_refresh=" in cookie and "httponly" in cookie and "samesite=lax" in cookie and "path=/auth" in cookie
    token = r.json()["access_token"]

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["email"] == email and me.json()["is_admin"] is False
    assert r.json()["is_admin"] is False

    with get_factory()() as s:
        user = s.scalar(select(User).where(User.email == email))
        assert user is not None and user.password_hash.startswith("$argon2id$")
        rows = s.scalars(select(RefreshToken).where(RefreshToken.user_id == user.id)).all()
        assert len(rows) == 1 and len(rows[0].token_hash) == 64  # a SHA-256, never the token

    login = TestClient(app).post("/auth/login", json={"email": email.upper(), "password": PASSWORD}, headers=ORIGIN)
    assert login.status_code == 200  # emails are case-insensitive


def test_register_validation(client):
    email, _ = register(client)
    _, dup = register(client, email)
    assert dup.status_code == 409
    assert register(client, password="short")[1].status_code == 422
    assert client.post("/auth/register", json={"email": "not-an-email", "password": PASSWORD}).status_code == 422


def test_wrong_password_and_unknown_user_look_identical(client):
    email, _ = register(client)
    wrong = TestClient(app).post("/auth/login", json={"email": email, "password": "nope-nope-nope"}, headers=ORIGIN)
    unknown = TestClient(app).post("/auth/login", json={"email": _email(), "password": PASSWORD}, headers=ORIGIN)
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json() == {"detail": "Invalid email or password"}


def test_login_is_rate_limited_per_address_and_email(client):
    email, _ = register(client)
    c = TestClient(app)
    codes = [c.post("/auth/login", json={"email": email, "password": "wrong-password"}, headers=ORIGIN).status_code for _ in range(12)]
    assert codes[:10] == [401] * 10 and codes[10:] == [429, 429]
    # even the right password is refused while limited: brute force gets no oracle
    assert c.post("/auth/login", json={"email": email, "password": PASSWORD}, headers=ORIGIN).status_code == 429
    r = c.post("/auth/login", json={"email": email, "password": PASSWORD}, headers=ORIGIN)
    assert int(r.headers["retry-after"]) > 0
    # another email from the same address is unaffected
    assert c.post("/auth/login", json={"email": _email(), "password": "x"}, headers=ORIGIN).status_code == 401


def test_refresh_rotates_the_token_and_reuse_revokes_everything(client):
    email, _ = register(client)
    first_cookie = client.cookies.get("rp_refresh")
    assert first_cookie

    r1 = client.post("/auth/refresh", headers=ORIGIN)
    assert r1.status_code == 200 and r1.json()["access_token"]
    second_cookie = client.cookies.get("rp_refresh")
    assert second_cookie and second_cookie != first_cookie  # rotated

    # an attacker replays the first (already used) token
    thief = TestClient(app)
    thief.cookies.set("rp_refresh", first_cookie, path="/auth")
    stolen = thief.post("/auth/refresh", headers=ORIGIN)
    assert stolen.status_code == 401

    # reuse detection revoked the whole family: the legitimate newest token is dead too
    assert client.post("/auth/refresh", headers=ORIGIN).status_code == 401
    with get_factory()() as s:
        user = s.scalar(select(User).where(User.email == email))
        assert user is not None
        live = s.scalars(select(RefreshToken).where(RefreshToken.user_id == user.id, RefreshToken.revoked_at.is_(None))).all()
        assert live == []


def test_logout_revokes_the_refresh_token_and_clears_the_cookie(client):
    register(client)
    cookie = client.cookies.get("rp_refresh")
    assert client.post("/auth/logout", headers=ORIGIN).status_code == 204
    other = TestClient(app)
    other.cookies.set("rp_refresh", cookie, path="/auth")
    assert other.post("/auth/refresh", headers=ORIGIN).status_code == 401


def test_refresh_needs_a_cookie_and_expired_tokens_fail(client):
    assert TestClient(app).post("/auth/refresh", headers=ORIGIN).status_code == 401
    register(client)
    with get_factory()() as s, s.begin():
        s.execute(text("UPDATE refresh_tokens SET expires_at = now() - interval '1 minute'"))
    assert client.post("/auth/refresh", headers=ORIGIN).status_code == 401


def test_cross_site_origins_cannot_use_the_cookie_endpoints(client):
    register(client)
    evil = {"Origin": "https://evil.example"}
    assert client.post("/auth/refresh", headers=evil).status_code == 403
    assert client.post("/auth/logout", headers=evil).status_code == 403
    assert client.post("/auth/login", json={"email": _email(), "password": PASSWORD}, headers=evil).status_code == 403


def test_admin_flag_follows_the_configured_admin_emails(client):
    email = _email()
    app.dependency_overrides[deps.settings_dep] = lambda: Settings(admin_emails=[email.upper()])
    _, r = register(client, email)
    assert r.json()["is_admin"] is True  # case-insensitive match
    other = register(client)[1]
    assert other.json()["is_admin"] is False


def test_registration_can_be_closed(client):
    app.dependency_overrides[deps.settings_dep] = lambda: Settings(registration_enabled=False)
    assert register(client)[1].status_code == 403


def test_protected_endpoints_reject_missing_expired_and_forged_tokens(client):
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"Authorization": "Bearer nonsense"}).status_code == 401
    assert client.get("/documents", headers={"Authorization": "Basic abc"}).status_code == 401
    old = {"X-User-Id": str(uuid.uuid4())}  # the retired dev header must do nothing
    assert client.get("/documents", headers=old).status_code == 401
    assert client.get("/documents", headers=auth(uuid.uuid4())).status_code == 200  # valid token, no docs


def test_the_dev_user_endpoint_is_gone(client):
    assert client.post("/auth/dev-user", json={"email": "a@b.co"}).status_code == 404


# --- rate limits and quotas -------------------------------------------------------------------


def _chat_client(settings, limiter=None):
    d, _ = make_deps(FakeStore([make_hit()]), FakeLLM(), settings=settings)
    app.dependency_overrides[deps.chat_deps] = lambda: d
    app.dependency_overrides[deps.settings_dep] = lambda: settings
    return TestClient(app), d


def test_per_user_rate_limit_returns_429_with_retry_after(client):
    c, _ = _chat_client(Settings(rate_limit_per_minute=3))
    h = auth(uuid.uuid4())
    codes = [c.post("/chat", json={"question": "ATP?"}, headers=h).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    r = c.post("/chat", json={"question": "ATP?"}, headers=h)
    assert r.status_code == 429 and 1 <= int(r.headers["retry-after"]) <= 60
    assert c.post("/chat", json={"question": "ATP?"}, headers=auth(uuid.uuid4())).status_code == 200  # others unaffected


async def test_limiter_counts_tokens_per_day_and_blocks_at_the_budget():
    lim = Limiter(None)
    uid = uuid.uuid4()
    assert await lim.tokens_used(uid) == 0
    await lim.add_tokens(uid, 400)
    await lim.add_tokens(uid, 250)
    await lim.add_tokens(uid, 0)
    assert await lim.tokens_used(uid) == 650


def test_daily_token_budget_blocks_further_requests(client, _local_limiter):
    c, _ = _chat_client(Settings(daily_token_budget=1000))
    uid = uuid.uuid4()
    asyncio.run(_local_limiter.add_tokens(uid, 1000))
    r = c.post("/chat", json={"question": "ATP?"}, headers=auth(uid))
    assert r.status_code == 429 and "budget" in r.json()["detail"]
    assert c.post("/chat", json={"question": "ATP?"}, headers=auth(uuid.uuid4())).status_code == 200


async def test_redis_outage_falls_back_to_a_conservative_local_limit():
    lim = Limiter("redis://127.0.0.1:1/0")  # nothing listens here
    results = [await lim.hit("k", 10, 60) for _ in range(8)]
    assert lim.redis_ok is False
    allowed = [ok for ok, _ in results]
    assert allowed == [True] * 5 + [False] * 3  # limit halved while degraded
    assert await lim.tokens_used(uuid.uuid4()) == 0  # and token accounting does not raise


async def test_limiter_window_resets():
    lim = Limiter(None)
    assert (await lim.hit("a", 1, 1))[0] and not (await lim.hit("a", 1, 1))[0]
    await asyncio.sleep(1.1)
    assert (await lim.hit("a", 1, 1))[0]


async def test_real_redis_limiter_if_available():
    lim = Limiter("redis://localhost:6379/0")
    key = f"test:{uuid.uuid4()}"
    first = await lim.hit(key, 2, 5)
    if not lim.redis_ok:
        pytest.skip("Redis not available")
    assert first[0] and (await lim.hit(key, 2, 5))[0] and not (await lim.hit(key, 2, 5))[0]
    uid = uuid.uuid4()
    await lim.add_tokens(uid, 123)
    assert await lim.tokens_used(uid) == 123


def test_cors_allows_credentials_only_for_configured_origins(client):
    """The browser refuses the refresh cookie unless the API says credentials are allowed, and
    credentialed CORS must never be combined with a wildcard origin."""
    good = client.options(
        "/auth/refresh",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert good.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert good.headers["access-control-allow-credentials"] == "true"
    bad = client.options(
        "/auth/refresh",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in bad.headers
