"""Aggregate metrics over query_logs (architecture.md section 11): latency percentiles per
stage, tokens and cost, cache hit rate by layer, refusal rate, error rate by type, ingestion."""

from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

_STAGE_ORDER = ["memory", "rewrite", "embed", "dense", "sparse", "rerank", "generation"]


def _f(x: Any) -> float | None:
    return None if x is None else round(float(x), 4)


def collect_metrics(s: Session, days: int = 7) -> dict[str, Any]:
    w = "created_at > now() - make_interval(days => :d)"
    p = {"d": days}
    t = s.execute(
        text(
            f"""select count(*) q, count(distinct user_id) users,
                  coalesce(sum(cost_usd),0) cost, coalesce(sum(tokens_in),0) tin,
                  coalesce(sum(tokens_out),0) tout, coalesce(sum(cache_read_tokens),0) cread,
                  percentile_cont(0.5) within group (order by latency_ms) p50,
                  percentile_cont(0.95) within group (order by latency_ms) p95,
                  count(*) filter (where gated) gated,
                  count(*) filter (where error is not null) errors
                from query_logs where {w}"""
        ),
        p,
    ).one()
    q = int(t.q)

    keys = [
        r[0]
        for r in s.execute(
            text(f"select distinct jsonb_object_keys(latency_breakdown) from query_logs where {w}"), p
        )
    ]
    ordered = [k for k in _STAGE_ORDER if k in keys] + sorted(k for k in keys if k not in _STAGE_ORDER)
    stages = {}
    for k in ordered:
        row = s.execute(
            text(
                f"""select percentile_cont(0.5) within group (order by (latency_breakdown->>:k)::numeric) p50,
                           percentile_cont(0.95) within group (order by (latency_breakdown->>:k)::numeric) p95,
                           count(latency_breakdown->>:k) n
                    from query_logs where {w} and latency_breakdown ? :k"""
            ),
            {**p, "k": k},
        ).one()
        stages[k] = {"p50_ms": _f(row.p50), "p95_ms": _f(row.p95), "samples": int(row.n)}

    cache = {r.cache_status: int(r.n) for r in s.execute(
        text(f"select cache_status, count(*) n from query_logs where {w} group by 1"), p)}
    hits = cache.get("exact", 0) + cache.get("semantic", 0)
    errors = {r.error: int(r.n) for r in s.execute(
        text(f"select error, count(*) n from query_logs where {w} and error is not null group by 1"), p)}
    modes = {r.mode: int(r.n) for r in s.execute(
        text(f"select mode, count(*) n from query_logs where {w} group by 1"), p)}
    daily = [
        {"day": r.d.date().isoformat(), "queries": int(r.n), "cost_usd": _f(r.c)}
        for r in s.execute(
            text(f"""select date_trunc('day', created_at) d, count(*) n, coalesce(sum(cost_usd),0) c
                     from query_logs where {w} group by 1 order by 1"""), p)
    ]
    degraded = {r.k: int(r.n) for r in s.execute(
        text(f"""select jsonb_array_elements_text(degraded) k, count(*) n
                 from query_logs where {w} group by 1"""), p)}

    docs = {r.status: int(r.n) for r in s.execute(text("select status, count(*) n from documents group by 1"))}
    per_page = s.execute(
        text(
            """select avg(extract(epoch from (b.last - b.first)) / nullif(d.page_count, 0)) sec_per_page
               from (select version_id, min(started_at) first, max(started_at) last
                     from ingestion_jobs group by 1) b
               join document_versions v on v.id = b.version_id
               join documents d on d.id = v.document_id
               where d.status = 'ready' and d.page_count > 0"""
        )
    ).scalar()
    pending = s.execute(text("select count(*) from outbox where dispatched_at is null")).scalar()
    ingested = sum(docs.get(k, 0) for k in ("ready", "failed"))

    return {
        "window_days": days,
        "queries": q,
        "active_users": int(t.users),
        "latency_ms": {"p50": _f(t.p50), "p95": _f(t.p95)},
        "stages": stages,
        "tokens": {"input": int(t.tin), "output": int(t.tout), "cache_read": int(t.cread)},
        "cost_usd": {
            "total": _f(t.cost),
            "per_query": _f(float(t.cost) / q) if q else None,
        },
        "cache": {"by_layer": cache, "hit_rate": round(hits / q, 4) if q else None},
        "refusal_rate": round(int(t.gated) / q, 4) if q else None,
        "error_rate": round(int(t.errors) / q, 4) if q else None,
        "errors_by_type": errors,
        "modes": modes,
        "degraded": degraded,
        "daily": daily,
        "ingestion": {
            "documents_by_status": docs,
            "failure_rate": round(docs.get("failed", 0) / ingested, 4) if ingested else None,
            "seconds_per_page": _f(per_page),
            "queue_depth": int(pending or 0) + docs.get("queued", 0),
        },
    }
