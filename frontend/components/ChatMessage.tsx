"use client";

import type { Citation, Closest } from "@/lib/api";
import { segmentAnswer } from "@/lib/citations";

export type ChatMsg = {
  id: number;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  streaming?: boolean;
  status?: string;
  error?: string;
  stopped?: boolean;
  closest?: Closest[];
  standalone?: string; // the rewritten search query, shown when it differs from the question
  question?: string;
};

type Props = {
  msg: ChatMsg;
  onCite: (c: Citation) => void;
  onClosest: (p: Closest) => void;
  activeChunk?: string;
};

export default function ChatMessage({ msg, onCite, onClosest, activeChunk }: Props) {
  if (msg.role === "user") {
    return (
      <div className="flex justify-end">
        <p className="max-w-[85%] whitespace-pre-wrap rounded-2xl bg-zinc-900 px-4 py-2 text-white dark:bg-zinc-100 dark:text-zinc-900">
          {msg.content}
        </p>
      </div>
    );
  }
  const segments = segmentAnswer(msg.content, msg.citations, !!msg.streaming);
  return (
    <div className="max-w-[90%] space-y-2">
      <p className="whitespace-pre-wrap leading-relaxed">
        {segments.map((s, i) =>
          s.kind === "text" ? (
            <span key={i}>{s.text}</span>
          ) : s.kind === "pending" ? (
            <span key={i} className="text-zinc-400">
              [·]
            </span>
          ) : (
            <button
              key={i}
              onClick={() => onCite(s.citation)}
              title={s.citation.quote}
              className={`mx-0.5 inline-flex items-center rounded-full border px-2 py-0 align-baseline text-xs hover:bg-yellow-100 dark:hover:bg-yellow-950 ${
                activeChunk === s.citation.chunk_id
                  ? "border-yellow-500 bg-yellow-100 dark:bg-yellow-950"
                  : "border-zinc-300 dark:border-zinc-700"
              }`}
            >
              p.{s.citation.page_start}
              {s.citation.page_end !== s.citation.page_start && `–${s.citation.page_end}`}
            </button>
          ),
        )}
        {msg.streaming && <span className="ml-0.5 inline-block animate-pulse">▍</span>}
      </p>
      {msg.streaming && !msg.content && msg.status && (
        <p className="text-sm text-zinc-500">{msg.status}…</p>
      )}
      {msg.citations.length > 0 && !msg.streaming && (
        <ul className="flex flex-wrap gap-1.5 text-xs text-zinc-500">
          {msg.citations.map((c) => (
            <li key={c.marker}>
              <button className="underline" onClick={() => onCite(c)}>
                {c.document_title}, p.{c.page_start}
              </button>
            </li>
          ))}
        </ul>
      )}
      {msg.closest && msg.closest.length > 0 && !msg.streaming && (
        <div className="rounded border border-zinc-200 p-2 text-xs dark:border-zinc-800">
          <p className="mb-1 text-zinc-500">Closest passages (too weak to answer from):</p>
          <ul className="space-y-1">
            {msg.closest.map((p, i) => (
              <li key={i}>
                <button className="text-left hover:underline" onClick={() => onClosest(p)}>
                  <span className="font-medium">
                    {p.document_title}, p.{p.page_start}
                  </span>
                  <span className="text-zinc-500"> — {p.quote.slice(0, 110)}…</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
      {msg.standalone && !msg.streaming && msg.standalone !== msg.question && (
        <p className="text-xs text-zinc-400">Searched for: {msg.standalone}</p>
      )}
      {msg.stopped && <p className="text-xs text-zinc-500">Stopped.</p>}
      {msg.error && <p className="text-sm text-red-600">{msg.error}</p>}
    </div>
  );
}
