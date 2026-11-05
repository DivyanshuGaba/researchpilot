import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import sessionmaker

from app.api import deps
from app.config import PipelineConfig, Settings
from app.db import get_factory, tenant_session
from app.main import app
from app.models import Message, QueryLog, User
from app.services.generation.llm_client import LLMUnavailable
from app.services.observability import QueryRecord, Tracer, config_hash, write_query_log

from .fakes import FakeLLM, auth
from .test_integration_db import PAGES, HashEmbedder, _db_ready, _make_pdf, _make_user

METRICS_URL = "postgresql+psycopg://rp_metrics:rp_metrics@localhost:5433/researchpilot"
db = pytest.mark.skipif(not _db_ready(), reason="Postgres not available")


@pytest.fixture(autouse=True)
def _clear():
    yield
    app.dependency_overrides.clear()


def test_config_hash_is_stable_and_changes_with_any_setting():
    a = config_hash(PipelineConfig())
    assert a == config_hash(PipelineConfig()) and len(a) == 16
    assert a != config_hash(PipelineConfig(top_k=7))
    assert a != config_hash(PipelineConfig(chunker="semantic"))
    assert a != config_hash(PipelineConfig(), mode="agent")


def _setup(tmp_path, llm, model="claude-sonnet-5-5"):
    from .fakes import OverlapReranker
    from .test_integration_db import ingest_version

    settings = Settings(upload_dir=str(tmp_path / "up"), llm_model=model)

    def enqueue(version_id: str, user_id: str) -> None:
        with tenant_session(uuid.UUID(user_id)) as s:
            ingest_version(s, uuid.UUID(version_id), HashEmbedder(), settings.pipeline)

    app.dependency_overrides[deps.settings_dep] = lambda: settings
    app.dependency_overrides[deps.enqueue_dep] = lambda: enqueue
    app.dependency_overrides[deps.embedder_dep] = lambda: HashEmbedder()
    app.dependency_overrides[deps.llm_dep] = lambda: llm
    app.dependency_overrides[deps.reranker_dep] = lambda: OverlapReranker()
    client = TestClient(app)
    user = _make_user()
    h = auth(user)
    pdf = tmp_path / "bio.pdf"
    _make_pdf(pdf, PAGES)
    assert client.post("/documents", headers=h, files={"file": ("bio.pdf", pdf.read_bytes())}).status_code == 202
    return client, h, user


@db
def test_every_answer_writes_a_query_log_with_stages_cost_and_masked_text(tmp_path):
    llm = FakeLLM("Mitochondria make ATP [c_1].")
    client, h, user = _setup(tmp_path, llm)
    q = "Email me at jane.doe@example.com or call 415-555-0132: how do mitochondria produce ATP?"
    body = client.post("/chat", headers=h, json={"question": q}).json()

    with tenant_session(user) as s:
        row = s.scalars(select(QueryLog)).one()
        assert row.mode == "qa" and row.cache_status == "miss" and not row.gated and row.error is None
        assert row.message_id == uuid.UUID(body["message_id"]) and len(row.config_hash) == 16
        assert row.latency_ms > 0 and {"embed", "dense", "generation"} <= set(row.latency_breakdown)
        assert row.retrieved_chunk_ids and float(row.top_score) > 0.5
        assert "jane.doe" not in row.rewritten_query and "415-555" not in row.rewritten_query
        assert "[EMAIL]" in row.rewritten_query and "[PHONE]" in row.rewritten_query
        assert row.tokens_in == 100 and row.tokens_out == 20  # the metered answer call
        # sonnet-5-5: $2 in / $10 out per million tokens
        assert float(row.cost_usd) == pytest.approx((100 * 2 + 20 * 10) / 1e6, rel=1e-6)
        msg = s.get(Message, uuid.UUID(body["message_id"]))
        assert msg is not None and float(msg.cost_usd) == pytest.approx(float(row.cost_usd))
        assert msg.latency_ms and msg.input_tokens == 100 and msg.output_tokens == 20


@db
def test_refusals_and_outages_are_logged_as_such(tmp_path):
    llm = FakeLLM()
    client, h, user = _setup(tmp_path, llm)
    client.post("/chat", headers=h, json={"question": "quantum chromodynamics gluon plasma lattice"})
    llm.error = LLMUnavailable("unavailable")
    assert client.post("/chat", headers=h, json={"question": "how do mitochondria produce ATP?"}).status_code == 503
    with tenant_session(user) as s:
        rows = {("gated" if r.gated else r.error or "ok"): r for r in s.scalars(select(QueryLog))}
    assert "gated" in rows and "unavailable" in rows
    assert rows["unavailable"].retrieved_chunk_ids  # what was retrieved before the model failed


@db
def test_a_broken_query_log_never_breaks_the_answer(tmp_path, monkeypatch):
    llm = FakeLLM("ATP [c_1]")
    client, h, _ = _setup(tmp_path, llm)

    def boom(*a, **k):
        raise RuntimeError("log database down")

    monkeypatch.setattr("app.services.chat_service.write_query_log", boom)
    r = client.post("/chat", headers=h, json={"question": "how do mitochondria produce ATP?"})
    assert r.status_code == 200 and r.json()["citations"]


def test_write_query_log_swallows_database_errors():
    def broken(_uid):
        raise RuntimeError("db down")

    write_query_log(broken, QueryRecord(user_id=uuid.uuid4(), mode="qa", config_hash="x" * 16))  # no raise


@db
def test_tenants_cannot_read_each_others_query_logs(tmp_path):
    client, h, _user = _setup(tmp_path, FakeLLM("ATP [c_1]"))
    client.post("/chat", headers=h, json={"question": "how do mitochondria produce ATP?"})
    other = _make_user()
    with tenant_session(other) as s:
        assert s.scalars(select(QueryLog)).all() == []
        assert s.execute(text("select count(*) from query_logs")).scalar() == 0


# --- admin metrics and per-user usage --------------------------------------------------------


@db
def test_admin_metrics_aggregate_across_tenants_and_are_admin_only(tmp_path):
    llm = FakeLLM("Mitochondria make ATP [c_1].")
    client, h, user = _setup(tmp_path, llm)
    for _ in range(2):
        client.post("/chat", headers=h, json={"question": "how do mitochondria produce ATP?"})
    client.post("/chat", headers=h, json={"question": "quantum chromodynamics gluon plasma lattice"})
    with get_factory()() as s:
        admin_email = s.get(User, user).email  # the same user acts as admin

    settings = Settings(admin_emails=[admin_email], metrics_database_url=METRICS_URL, llm_model="claude-sonnet-5-5")
    app.dependency_overrides[deps.settings_dep] = lambda: settings
    m = client.get("/admin/metrics?days=1", headers=h)
    assert m.status_code == 200, m.text
    data = m.json()
    assert data["queries"] >= 3 and data["active_users"] >= 1
    assert data["latency_ms"]["p95"] >= data["latency_ms"]["p50"] > 0
    assert {"embed", "dense", "generation"} <= set(data["stages"])
    assert data["stages"]["dense"]["p95_ms"] >= data["stages"]["dense"]["p50_ms"]
    assert data["cost_usd"]["total"] > 0 and data["cost_usd"]["per_query"] > 0
    assert data["cache"]["by_layer"]["miss"] >= 3 and data["cache"]["hit_rate"] == 0
    assert 0 < data["refusal_rate"] < 1  # one of the three was refused
    assert data["modes"]["qa"] >= 3 and data["daily"][-1]["queries"] >= 3
    assert data["ingestion"]["documents_by_status"].get("ready", 0) >= 1

    stranger = auth(_make_user())  # a signed-in user who is not an admin
    assert client.get("/admin/metrics", headers=stranger).status_code == 403
    assert TestClient(app).get("/admin/metrics").status_code == 401


@db
def test_metrics_role_is_read_only_and_the_app_role_is_row_isolated():
    eng = create_engine(METRICS_URL)
    with sessionmaker(eng)() as s:
        assert s.execute(text("select count(*) from query_logs")).scalar() is not None  # sees all tenants
        with pytest.raises(DBAPIError, match="permission denied"):
            s.execute(text("delete from query_logs"))
        s.rollback()
        with pytest.raises(DBAPIError, match="permission denied"):
            s.execute(text("update users set email = email"))
        s.rollback()
    with get_factory()() as s:  # the app role has no tenant set: sees nothing
        assert s.execute(text("select count(*) from query_logs")).scalar() == 0


@db
def test_usage_endpoint_returns_only_my_costs(tmp_path):
    client, h, _ = _setup(tmp_path, FakeLLM("ATP [c_1]"))
    client.post("/chat", headers=h, json={"question": "how do mitochondria produce ATP?"})
    mine = client.get("/usage", headers=h).json()
    assert mine["queries"] == 1 and mine["cost_usd"] > 0
    assert mine["by_conversation"][0]["queries"] == 1
    other = client.get("/usage", headers=auth(_make_user())).json()
    assert other["queries"] == 0 and other["cost_usd"] == 0 and other["by_conversation"] == []


# --- Langfuse exporter -----------------------------------------------------------------------


def _rec():
    return QueryRecord(
        user_id=uuid.uuid4(), mode="qa", config_hash="abc", latency_ms=120, tokens_in=500,
        tokens_out=80, cost_usd=0.0012, latency_breakdown={"dense": 12, "rerank": 80},
    )


def test_tracer_is_off_without_keys_and_masks_pii_when_on():
    assert Tracer(Settings()).enabled is False
    Tracer(Settings()).trace(_rec(), "q", "a", "m")  # no-op, no event loop needed

    s = Settings(langfuse_public_key="pk", langfuse_secret_key="sk", langfuse_host="https://lf.example")
    t = Tracer(s)
    assert t.enabled
    batch = t.build_batch(_rec(), "mail me at a.b@example.com, key sk-ant-api03-AbCdEfGhIjKlMnOp", "ok", "claude-sonnet-5-5")
    kinds = [e["type"] for e in batch["batch"]]
    assert kinds[0] == "trace-create" and kinds.count("span-create") == 2 and kinds[-1] == "generation-create"
    dumped = str(batch)
    assert "a.b@example.com" not in dumped and "AbCdEfGh" not in dumped
    assert "[EMAIL]" in dumped and "[REDACTED_KEY]" in dumped
    gen = batch["batch"][-1]["body"]
    assert gen["usage"]["input"] == 500 and gen["usage"]["totalCost"] == 0.0012


async def test_tracer_export_failures_are_ignored():
    s = Settings(langfuse_public_key="pk", langfuse_secret_key="sk", langfuse_host="http://127.0.0.1:1")
    await Tracer(s)._send(_rec(), "q", "a", "m")  # nothing listens; must not raise
