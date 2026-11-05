import json
import uuid

import pytest
from sqlalchemy import select

from app.api import deps
from app.config import Settings
from app.db import tenant_session
from app.main import app
from app.models import QueryLog
from app.services.cache import (
    CachedEmbedder,
    RedisCache,
    decode_vec,
    docset_hash,
    encode_vec,
    normalize_query,
)
from app.services.observability import pipeline_hash

from .fakes import FakeLLM
from .test_integration_db import PAGES, _db_ready, _make_pdf
from .test_observability import _setup

REDIS = "redis://localhost:6379/0"


def _redis_up() -> bool:
    try:
        import redis

        redis.Redis.from_url(REDIS, socket_connect_timeout=0.3).ping()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_stack = pytest.mark.skipif(not (_db_ready() and _redis_up()), reason="Postgres and Redis needed")


class DictCache(RedisCache):
    """RedisCache semantics on a dict: unit tests without a server."""

    def __init__(self):
        super().__init__("redis://unused", 60, True)
        self.kv: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}

    def _call(self, fn, default=None):  # pragma: no cover - replaced by explicit methods below
        return default

    def get_embeddings(self, keys):
        return [decode_vec(self.kv[k]) if k in self.kv else None for k in keys]

    def set_embeddings(self, items):
        self.kv.update({k: encode_vec(v) for k, v in items.items()})

    def get_exact(self, key):
        return json.loads(self.kv[key]) if key in self.kv else None

    def set_exact(self, key, payload):
        self.kv[key] = json.dumps(payload)

    def semantic_entries(self, key):
        return [(decode_vec(d["v"]), d["p"]) for d in map(json.loads, self.lists.get(key, []))]

    def semantic_add(self, key, vec, payload):
        self.lists.setdefault(key, []).insert(0, json.dumps({"v": encode_vec(vec), "p": payload}))
        del self.lists[key][50:]


# --- unit ---------------------------------------------------------------------------------------


def test_vector_encoding_roundtrips_and_queries_normalize():
    v = [0.1, -0.25, 0.5, 0.0]
    assert decode_vec(encode_vec(v)) == pytest.approx(v)
    assert normalize_query("  What IS   ATP?? ") == normalize_query("what is atp")
    assert normalize_query("what is atp") != normalize_query("what is adp")


def test_keys_are_scoped_by_user_document_set_config_mode_and_query():
    c = RedisCache("redis://x", 60, True)
    u1, u2 = uuid.uuid4(), uuid.uuid4()
    base = c.exact_key(u1, "docset-A", "cfg", "qa", "what is atp")
    assert base == c.exact_key(u1, "docset-A", "cfg", "qa", "What is ATP?")  # trivial variants share
    for other in (
        c.exact_key(u2, "docset-A", "cfg", "qa", "what is atp"),  # another user
        c.exact_key(u1, "docset-B", "cfg", "qa", "what is atp"),  # documents changed
        c.exact_key(u1, "docset-A", "cfg2", "qa", "what is atp"),  # settings changed
        c.exact_key(u1, "docset-A", "cfg", "compare", "what is atp"),  # another mode
        c.exact_key(u1, "docset-A", "cfg", "qa", "what is adp"),
    ):
        assert other != base
    assert c.semantic_key(u1, "A", "cfg", "qa") != c.semantic_key(u2, "A", "cfg", "qa")


def test_pipeline_hash_covers_everything_that_changes_an_answer():
    base = pipeline_hash(Settings(), "qa")
    assert base == pipeline_hash(Settings(), "qa")
    for changed in (
        Settings(llm_model="other-model"), Settings(retrieval_min_score=0.2),
        Settings(embedding_model="other"), Settings(reranker="none"),
    ):
        assert pipeline_hash(changed, "qa") != base
    assert pipeline_hash(Settings(), "compare") != base


def test_semantic_lookup_uses_the_threshold_and_prefers_the_closest():
    c = DictCache()
    key = "sem-key"
    c.semantic_add(key, [1.0, 0.0], {"answer": "far"})
    c.semantic_add(key, [0.96, 0.28], {"answer": "near"})  # unit-ish vector, cosine .96 with [1,0]
    hit = c.lookup("no-exact", key, [1.0, 0.0], 0.95)
    assert hit and hit[0] == "semantic" and hit[1]["answer"] == "far" and hit[2] == pytest.approx(1.0)
    assert c.lookup("no-exact", key, [0.0, 1.0], 0.95) is None  # orthogonal: no hit
    just_below = [0.9, 0.436]  # cosine with [1,0] about 0.9 -> below 0.95
    assert c.lookup("no-exact", "empty-key", just_below, 0.95) is None
    c.set_exact("e", {"answer": "exact"})
    assert c.lookup("e", key, [0.0, 1.0], 0.95)[0] == "exact"  # exact wins and needs no vector match


def test_a_disabled_or_unreachable_cache_is_a_silent_no_op():
    off = RedisCache("redis://127.0.0.1:1/0", 60, enabled=False)
    assert off.get_exact("k") is None and off.lookup("a", "b", [1.0], 0.9) is None
    off.set_exact("k", {"a": 1})  # no raise

    down = RedisCache("redis://127.0.0.1:1/0", 60, enabled=True)
    assert down.get_exact("k") is None and down.semantic_entries("k") == []
    down.store("a", "b", [1.0], {"x": 1})
    assert down._down_until > 0  # remembered, so later calls do not each wait for a timeout
    calls = []
    down._client = lambda: calls.append(1)  # type: ignore[method-assign]
    down.get_exact("k")
    assert calls == []  # backed off: the client is not even created


class CountingEmbedder:
    model_id, dim = "m", 3

    def __init__(self):
        self.q, self.d = [], []

    def embed_query(self, t):
        self.q.append(t)
        return [float(len(t)), 0.0, 1.0]

    def embed_documents(self, ts):
        self.d.append(list(ts))
        return [[float(len(t)), 1.0, 0.0] for t in ts]


def test_embedding_cache_avoids_recomputing_queries_and_passages():
    inner, cache = CountingEmbedder(), DictCache()
    emb = CachedEmbedder(inner, cache)
    a = emb.embed_query("what is atp")
    assert emb.embed_query("what is atp") == a and inner.q == ["what is atp"]  # computed once
    docs = emb.embed_documents(["one", "two"])
    again = emb.embed_documents(["one", "two", "three"])  # only the new text is computed
    assert inner.d == [["one", "two"], ["three"]] and again[:2] == docs
    # a query and a passage with the same text must not share a cache entry (different prefixes)
    assert emb.embed_documents(["what is atp"])[0] != a


# --- integration: real Redis and Postgres -----------------------------------------------------


@pytest.fixture
def real_cache(monkeypatch):
    monkeypatch.setattr(
        "app.api.deps._shared_cache", lambda url, ttl, enabled: RedisCache(url, ttl, enabled)
    )


Q = "how do mitochondria produce ATP?"


def _ask(client, h, q=Q, **kw):
    return client.post("/chat", headers=h, json={"question": q, **kw}).json()


@needs_stack
def test_second_identical_question_is_served_from_the_exact_cache(tmp_path, real_cache):
    llm = FakeLLM("Mitochondria make ATP [c_1].")
    client, h, user = _setup(tmp_path, llm, cache_ttl_seconds=60)
    first = _ask(client, h)
    assert first["meta"]["cache_status"] == "miss" and len(llm.requests) == 1
    second = _ask(client, h, "  How do mitochondria produce ATP  ")  # trivially different text
    assert second["meta"]["cache_status"] == "exact" and len(llm.requests) == 1  # no model call
    assert second["answer"] == first["answer"] and second["citations"] == first["citations"]
    assert second["citations"][0]["page_start"] >= 1 and second["message_id"] != first["message_id"]
    assert "retrieve" not in second["meta"]["timings_ms"] and "dense" not in second["meta"]["timings_ms"]
    with tenant_session(user) as s:
        rows = list(s.scalars(select(QueryLog).order_by(QueryLog.created_at)))
        assert [r.cache_status for r in rows] == ["miss", "exact"] and rows[1].cost_usd == 0


@needs_stack
def test_paraphrases_hit_the_semantic_cache_and_unrelated_questions_do_not(tmp_path, real_cache):
    llm = FakeLLM("Mitochondria make ATP [c_1].")
    client, h, _ = _setup(tmp_path, llm, cache_ttl_seconds=60, semantic_cache_threshold=0.9)
    _ask(client, h)
    near = _ask(client, h, "how do mitochondria produce ATP please")
    assert near["meta"]["cache_status"] == "semantic" and len(llm.requests) == 1
    far = _ask(client, h, "what does photosynthesis make from light")
    assert far["meta"]["cache_status"] == "miss" and len(llm.requests) == 2


@needs_stack
def test_a_stricter_threshold_turns_the_same_paraphrase_into_a_miss(tmp_path, real_cache):
    llm = FakeLLM("Mitochondria make ATP [c_1].")
    client, h, _ = _setup(tmp_path, llm, cache_ttl_seconds=60, semantic_cache_threshold=0.99)
    _ask(client, h)
    assert _ask(client, h, "how do mitochondria produce ATP please")["meta"]["cache_status"] == "miss"


@needs_stack
def test_changing_the_documents_invalidates_cached_answers(tmp_path, real_cache):
    llm = FakeLLM("Mitochondria make ATP [c_1].")
    client, h, user = _setup(tmp_path, llm, cache_ttl_seconds=60)
    _ask(client, h)
    assert _ask(client, h)["meta"]["cache_status"] == "exact"
    with tenant_session(user) as s:
        before = docset_hash(s, user, None)
    pdf = tmp_path / "more.pdf"
    _make_pdf(pdf, ["A second document about mitochondria and ATP synthase."])
    client.post("/documents", headers=h, files={"file": ("more.pdf", pdf.read_bytes())})
    with tenant_session(user) as s:
        assert docset_hash(s, user, None) != before  # the document set has a new version
    assert _ask(client, h)["meta"]["cache_status"] == "miss"  # so the old answer is not reused


@needs_stack
def test_other_users_never_see_cached_answers(tmp_path, real_cache):
    from .fakes import auth
    from .test_integration_db import _make_user

    llm = FakeLLM("Mitochondria make ATP [c_1].")
    client, h, _ = _setup(tmp_path, llm, cache_ttl_seconds=60)
    _ask(client, h)
    stranger = auth(_make_user())
    body = _ask(client, stranger)
    assert body["meta"]["cache_status"] == "miss" and "don't seem to cover" in body["answer"]


@needs_stack
def test_agent_degraded_and_failed_answers_are_never_cached(tmp_path, real_cache):
    from .fakes import text_turn, tool_turn

    llm = FakeLLM()
    client, h, _ = _setup(tmp_path, llm, cache_ttl_seconds=60)
    llm.script = [tool_turn(("search_documents", {"query": "ATP"})), text_turn("a [c_1]")] * 2
    for _ in range(2):
        out = _ask(client, h, mode="agent")
        assert out["meta"]["mode"] == "agent" and "cache_status" not in out["meta"]
    assert len([r for r in llm.converse_requests if r.tools]) >= 4  # the model ran both times

    # degraded run (no reranker): answered, but not stored
    app.dependency_overrides[deps.reranker_dep] = lambda: None
    llm2 = FakeLLM("ATP [c_1].")
    app.dependency_overrides[deps.llm_dep] = lambda: llm2
    _ask(client, h, "does ATP degrade?")
    again = _ask(client, h, "does ATP degrade?")
    assert again["meta"]["cache_status"] == "miss" and len(llm2.requests) == 2

    # an outage answer is not stored either
    from app.services.generation.llm_client import LLMUnavailable

    llm3 = FakeLLM()
    llm3.error = LLMUnavailable("unavailable")
    app.dependency_overrides[deps.llm_dep] = lambda: llm3
    assert client.post("/chat", headers=h, json={"question": "outage question atp"}).status_code == 503
    llm3.error = None
    llm3.text = "fine [c_1]"
    assert _ask(client, h, "outage question atp")["meta"]["cache_status"] == "miss"


@needs_stack
def test_changing_a_setting_that_affects_answers_changes_the_cache_namespace(tmp_path, real_cache):
    llm = FakeLLM("Mitochondria make ATP [c_1].")
    client, h, _ = _setup(tmp_path, llm, cache_ttl_seconds=60)
    _ask(client, h)
    assert _ask(client, h)["meta"]["cache_status"] == "exact"
    stricter = Settings(upload_dir=str(tmp_path / "up"), llm_model="claude-sonnet-5-5",
                        cache_ttl_seconds=60, retrieval_min_score=0.5)
    app.dependency_overrides[deps.settings_dep] = lambda: stricter
    assert _ask(client, h)["meta"]["cache_status"] == "miss"


@needs_stack
def test_refusals_are_cached_with_their_closest_passages(tmp_path, real_cache):
    llm = FakeLLM()
    client, h, _ = _setup(tmp_path, llm, cache_ttl_seconds=60)
    q = "quantum chromodynamics gluon plasma lattice"
    first = _ask(client, h, q)
    second = _ask(client, h, q)
    assert "don't seem to cover" in first["answer"] and first["closest"]
    assert second["meta"]["cache_status"] == "exact" and second["closest"] == first["closest"]
    assert llm.requests == []


@needs_stack
def test_streaming_replays_a_cached_answer_with_citations(tmp_path, real_cache):
    llm = FakeLLM("Mitochondria make ATP [c_1].")
    client, h, _ = _setup(tmp_path, llm, cache_ttl_seconds=60)
    _ask(client, h)  # warm the cache
    with client.stream("POST", "/chat/stream", headers=h, json={"question": Q}) as r:
        body = "".join(r.iter_text())
    assert body.index("event: token") < body.index("event: citation") < body.index("event: done")
    assert '"cache_status": "exact"' in body and len(llm.requests) == 1


@needs_stack
def test_a_redis_outage_does_not_break_chat(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.api.deps._shared_cache", lambda url, ttl, enabled: RedisCache("redis://127.0.0.1:1/0", ttl, True)
    )
    llm = FakeLLM("Mitochondria make ATP [c_1].")
    client, h, _ = _setup(tmp_path, llm)
    a, b = _ask(client, h), _ask(client, h)
    assert a["citations"] and b["citations"] and len(llm.requests) == 2  # served normally, uncached


def test_unused_imports_guard():
    assert PAGES and _make_pdf
