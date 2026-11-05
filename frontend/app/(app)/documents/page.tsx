"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { addUrl, listDocuments, uploadDocument, type Doc } from "@/lib/api";

type Upload = { id: number; name: string; fraction: number; error?: string };

const BADGE: Record<Doc["status"], string> = {
  queued: "bg-zinc-200 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300",
  processing: "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300",
  ready: "bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300",
  failed: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
};

const inFlight = (d: Doc) => d.status === "queued" || d.status === "processing";

export default function DocumentsPage() {
  const [docs, setDocs] = useState<Doc[] | null>(null);
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [url, setUrl] = useState("");
  const nextId = useRef(0);

  const refresh = useCallback(async () => {
    try {
      setDocs(await listDocuments());
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    let live = true;
    listDocuments()
      .then((d) => live && setDocs(d))
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, []);

  // Poll only while something is still being processed.
  const needsPolling = docs?.some(inFlight) ?? false;
  useEffect(() => {
    if (!needsPolling) return;
    const t = setInterval(() => void refresh(), 2000);
    return () => clearInterval(t);
  }, [needsPolling, refresh]);

  async function onFiles(files: FileList | null) {
    if (!files) return;
    for (const file of Array.from(files)) {
      const id = nextId.current++;
      const patch = (p: Partial<Upload>) =>
        setUploads((u) => u.map((x) => (x.id === id ? { ...x, ...p } : x)));
      setUploads((u) => [...u, { id, name: file.name, fraction: 0 }]);
      try {
        await uploadDocument(file, (fraction) => patch({ fraction }));
        setUploads((u) => u.filter((x) => x.id !== id));
        await refresh();
      } catch (e) {
        patch({ error: (e as Error).message });
      }
    }
  }

  async function onUrl(e: React.FormEvent) {
    e.preventDefault();
    try {
      await addUrl(url.trim());
      setUrl("");
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <div className="mx-auto h-full max-w-3xl space-y-6 overflow-auto p-4">
      <section
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          void onFiles(e.dataTransfer.files);
        }}
        className="rounded-lg border-2 border-dashed border-zinc-300 p-6 text-center dark:border-zinc-700"
      >
        <p className="mb-3 text-sm text-zinc-500">Drop PDF or DOCX files here, or</p>
        <label className="cursor-pointer rounded bg-zinc-900 px-3 py-1.5 text-sm text-white dark:bg-zinc-100 dark:text-zinc-900">
          Choose files
          <input
            type="file"
            multiple
            accept=".pdf,.docx,application/pdf"
            className="hidden"
            onChange={(e) => {
              void onFiles(e.target.files);
              e.target.value = "";
            }}
          />
        </label>
        <form onSubmit={onUrl} className="mt-4 flex gap-2">
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="…or paste a web page URL"
            type="url"
            className="flex-1 rounded border border-zinc-300 bg-transparent px-3 py-1.5 text-sm dark:border-zinc-700"
          />
          <button
            disabled={!url}
            className="rounded border border-zinc-300 px-3 py-1.5 text-sm disabled:opacity-50 dark:border-zinc-700"
          >
            Add URL
          </button>
        </form>
      </section>

      {uploads.length > 0 && (
        <ul className="space-y-2">
          {uploads.map((u) => (
            <li key={u.id} className="text-sm">
              <div className="flex justify-between">
                <span className="truncate">{u.name}</span>
                <span className={u.error ? "text-red-600" : "text-zinc-500"}>
                  {u.error ?? `${Math.round(u.fraction * 100)}%`}
                </span>
              </div>
              <div className="mt-1 h-1.5 overflow-hidden rounded bg-zinc-200 dark:bg-zinc-800">
                <div
                  className={`h-full ${u.error ? "bg-red-500" : "bg-blue-500"}`}
                  style={{ width: `${u.error ? 100 : u.fraction * 100}%` }}
                />
              </div>
            </li>
          ))}
        </ul>
      )}

      {error && <p className="text-sm text-red-600">{error}</p>}

      <section>
        <h2 className="mb-2 text-sm font-medium text-zinc-500">Your documents</h2>
        {docs === null && !error && <p className="text-sm text-zinc-500">Loading…</p>}
        {docs?.length === 0 && (
          <p className="text-sm text-zinc-500">Nothing here yet. Upload a document to begin.</p>
        )}
        <ul className="divide-y divide-zinc-200 dark:divide-zinc-800">
          {docs?.map((d) => (
            <li key={d.id} className="flex items-center justify-between gap-3 py-3">
              <div className="min-w-0">
                <p className="truncate font-medium">{d.title}</p>
                <p className="text-xs text-zinc-500">
                  {d.source_type.toUpperCase()}
                  {d.page_count > 0 && ` · ${d.page_count} page${d.page_count === 1 ? "" : "s"}`}
                  {d.status === "processing" && d.stage && ` · ${d.stage}…`}
                </p>
              </div>
              <span className={`rounded-full px-2.5 py-0.5 text-xs ${BADGE[d.status]}`}>
                {d.status}
              </span>
            </li>
          ))}
        </ul>
        {docs?.some((d) => d.status === "ready") && (
          <Link href="/chat" className="mt-4 inline-block text-sm underline">
            Chat with your documents →
          </Link>
        )}
      </section>
    </div>
  );
}
