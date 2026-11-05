"use client";

import { useEffect, useState } from "react";
import { fetchMetrics, type Metrics } from "@/lib/api";
import { ApiError } from "@/lib/session";

const pct = (v: number | null) => (v === null ? "n/a" : `${(v * 100).toFixed(1)}%`);
const ms = (v: number | null) => (v === null ? "n/a" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`);
const usd = (v: number | null) => (v === null ? "n/a" : `$${v < 0.01 ? v.toFixed(5) : v.toFixed(3)}`);

function Card({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
      <p className="text-xs text-zinc-500">{label}</p>
      <p className="text-xl font-semibold tabular-nums">{value}</p>
      {hint && <p className="text-xs text-zinc-400">{hint}</p>}
    </div>
  );
}

function Bar({ value, max, label }: { value: number; max: number; label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-28 shrink-0 text-zinc-500">{label}</span>
      <div className="h-2 flex-1 rounded bg-zinc-100 dark:bg-zinc-900">
        <div className="h-2 rounded bg-blue-500" style={{ width: `${max ? Math.max(2, (value / max) * 100) : 0}%` }} />
      </div>
      <span className="w-16 text-right tabular-nums">{value}</span>
    </div>
  );
}

export default function MetricsPage() {
  const [days, setDays] = useState(7);
  const [data, setData] = useState<Metrics | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    let live = true;
    fetchMetrics(days)
      .then((m) => live && (setData(m), setError(null)))
      .catch((e: ApiError) => live && setError(e));
    return () => {
      live = false;
    };
  }, [days]);

  if (error?.status === 403) return <p className="p-6 text-zinc-500">Metrics are for administrators only.</p>;
  if (error) return <p className="p-6 text-red-600">{error.message}</p>;
  if (!data) return <p className="p-6 text-sm text-zinc-500">Loading…</p>;

  const stageMax = Math.max(1, ...Object.values(data.stages).map((s) => s.p95_ms ?? 0));
  const dayMax = Math.max(1, ...data.daily.map((d) => d.queries));
  const cacheTotal = Object.values(data.cache.by_layer).reduce((a, b) => a + b, 0);

  return (
    <div className="h-full overflow-auto p-4">
      <div className="mx-auto max-w-4xl space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-lg font-semibold">Metrics</h1>
          <label className="text-sm">
            <span className="mr-2 text-zinc-500">Window</span>
            <select
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className="rounded border border-zinc-300 bg-transparent px-2 py-1 dark:border-zinc-700"
            >
              {[1, 7, 30, 90].map((d) => (
                <option key={d} value={d} className="text-black">
                  {d} day{d === 1 ? "" : "s"}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Card label="Queries" value={String(data.queries)} hint={`${data.active_users} active users`} />
          <Card label="Latency p50 / p95" value={`${ms(data.latency_ms.p50)} / ${ms(data.latency_ms.p95)}`} />
          <Card label="Cost per query" value={usd(data.cost_usd.per_query)} hint={`${usd(data.cost_usd.total)} total`} />
          <Card label="Cache hit rate" value={pct(data.cache.hit_rate)} />
          <Card label="Refusal rate" value={pct(data.refusal_rate)} hint="blocked by the confidence gate" />
          <Card label="Error rate" value={pct(data.error_rate)} />
          <Card label="Tokens in / out" value={`${data.tokens.input.toLocaleString()} / ${data.tokens.output.toLocaleString()}`} />
          <Card label="Ingestion queue" value={String(data.ingestion.queue_depth)} hint={`failure rate ${pct(data.ingestion.failure_rate)}`} />
        </div>

        <section>
          <h2 className="mb-2 text-sm font-medium text-zinc-500">Latency by stage (p95, with p50 in the table)</h2>
          <div className="space-y-1.5">
            {Object.entries(data.stages).map(([stage, s]) => (
              <Bar key={stage} label={stage} value={Math.round(s.p95_ms ?? 0)} max={stageMax} />
            ))}
            {Object.keys(data.stages).length === 0 && <p className="text-sm text-zinc-500">No data in this window.</p>}
          </div>
          <table className="mt-3 w-full text-left text-xs">
            <thead className="text-zinc-500">
              <tr>
                <th className="py-1 font-normal">Stage</th>
                <th className="font-normal">p50</th>
                <th className="font-normal">p95</th>
                <th className="font-normal">samples</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(data.stages).map(([stage, s]) => (
                <tr key={stage} className="border-t border-zinc-100 dark:border-zinc-900">
                  <td className="py-1">{stage}</td>
                  <td>{ms(s.p50_ms)}</td>
                  <td>{ms(s.p95_ms)}</td>
                  <td>{s.samples}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <div className="grid gap-6 sm:grid-cols-2">
          <section>
            <h2 className="mb-2 text-sm font-medium text-zinc-500">Cache layers</h2>
            <div className="space-y-1.5">
              {Object.entries(data.cache.by_layer).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={cacheTotal} />
              ))}
            </div>
          </section>
          <section>
            <h2 className="mb-2 text-sm font-medium text-zinc-500">Errors and degraded runs</h2>
            {Object.keys(data.errors_by_type).length + Object.keys(data.degraded).length === 0 ? (
              <p className="text-sm text-zinc-500">None in this window.</p>
            ) : (
              <ul className="space-y-1 text-sm">
                {Object.entries(data.errors_by_type).map(([k, v]) => (
                  <li key={k}>
                    <span className="font-mono">{k}</span>: {v}
                  </li>
                ))}
                {Object.entries(data.degraded).map(([k, v]) => (
                  <li key={k} className="text-zinc-500">
                    degraded <span className="font-mono">{k}</span>: {v}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>

        <section>
          <h2 className="mb-2 text-sm font-medium text-zinc-500">Queries per day</h2>
          <div className="space-y-1.5">
            {data.daily.map((d) => (
              <Bar key={d.day} label={d.day} value={d.queries} max={dayMax} />
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
