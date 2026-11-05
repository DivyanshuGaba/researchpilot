"use client";

import { useEffect, useRef, useState } from "react";
import ChatMessage, { type ChatMsg } from "@/components/ChatMessage";
import PdfPanel from "@/components/PdfPanel";
import {
  listDocuments,
  streamChat,
  type ChatEvent,
  type Citation,
  type Closest,
  type Doc,
} from "@/lib/api";

type Viewer = { citation: Citation; nonce: number };

const STAGE_LABEL: Record<string, string> = {
  rewriting: "Understanding your question",
  searching: "Searching your documents",
  reranking: "Ranking passages",
  writing: "Writing",
};

export default function ChatPage() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [scope, setScope] = useState<Set<string>>(new Set());
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [viewer, setViewer] = useState<Viewer | null>(null);
  const [compare, setCompare] = useState(false);
  const conversationId = useRef<string | undefined>(undefined);
  const abort = useRef<AbortController | null>(null);
  const nextId = useRef(0);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listDocuments()
      .then((d) => setDocs(d.filter((x) => x.status === "ready")))
      .catch(() => setDocs([]));
  }, []);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const patchLast = (fn: (m: ChatMsg) => ChatMsg) =>
    setMessages((ms) => ms.map((m, i) => (i === ms.length - 1 ? fn(m) : m)));

  function onEvent(e: ChatEvent) {
    switch (e.type) {
      case "status":
        patchLast((m) => ({ ...m, status: STAGE_LABEL[e.stage] ?? "Working" }));
        break;
      case "closest":
        patchLast((m) => ({ ...m, closest: e.passages }));
        break;
      case "token":
        patchLast((m) => ({ ...m, content: m.content + e.text }));
        break;
      case "citation":
        patchLast((m) => ({ ...m, citations: [...m.citations, e.citation] }));
        break;
      case "done":
        conversationId.current = e.conversationId;
        // The server's answer is authoritative: invalid citation markers are already removed.
        patchLast((m) => ({
          ...m,
          content: e.answer,
          streaming: false,
          closest: e.closest.length ? e.closest : m.closest,
          standalone: e.standaloneQuery,
        }));
        break;
      case "error":
        patchLast((m) => ({ ...m, error: e.message, streaming: false }));
        break;
    }
  }

  async function send() {
    const question = input.trim();
    if (!question || busy) return;
    setInput("");
    setBusy(true);
    setMessages((ms) => [
      ...ms,
      { id: nextId.current++, role: "user", content: question, citations: [] },
      {
        id: nextId.current++,
        role: "assistant",
        content: "",
        citations: [],
        streaming: true,
        status: "Searching your documents",
        question,
      },
    ]);
    const controller = new AbortController();
    abort.current = controller;
    try {
      await streamChat(
        {
          question,
          conversation_id: conversationId.current,
          document_ids: scope.size ? [...scope] : undefined,
          mode: compareOn ? "compare" : "qa",
        },
        onEvent,
        controller.signal,
      );
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        patchLast((m) => ({ ...m, streaming: false, stopped: true }));
      } else {
        patchLast((m) => ({ ...m, streaming: false, error: "Connection lost. Please try again." }));
      }
    } finally {
      patchLast((m) => (m.streaming ? { ...m, streaming: false } : m));
      setBusy(false);
      abort.current = null;
    }
  }

  // Compare needs at least two selected documents; it silently reverts to Q&A below that.
  const compareOn = compare && scope.size >= 2;

  const toggle = (id: string) =>
    setScope((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });

  const openCitation = (c: Citation) => setViewer((v) => ({ citation: c, nonce: (v?.nonce ?? 0) + 1 }));
  const openClosest = (p: Closest) =>
    openCitation({
      marker: "",
      chunk_id: "",
      document_id: p.document_id,
      document_title: p.document_title,
      page_start: p.page_start,
      page_end: p.page_start,
      quote: p.quote,
      bbox_list: p.bbox_list,
    });

  return (
    <div className="flex h-full">
      <section className="flex min-w-0 flex-1 flex-col">
        {docs.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 border-b border-zinc-200 px-4 py-2 text-xs dark:border-zinc-800">
            <span className="text-zinc-500">Search in:</span>
            {docs.map((d) => (
              <label
                key={d.id}
                className={`cursor-pointer rounded-full border px-2.5 py-0.5 ${
                  scope.has(d.id)
                    ? "border-zinc-900 bg-zinc-900 text-white dark:border-zinc-100 dark:bg-zinc-100 dark:text-zinc-900"
                    : "border-zinc-300 dark:border-zinc-700"
                }`}
              >
                <input type="checkbox" className="hidden" checked={scope.has(d.id)} onChange={() => toggle(d.id)} />
                {d.title}
              </label>
            ))}
            {scope.size === 0 && <span className="text-zinc-500">(all documents)</span>}
            <label
              className={`ml-auto flex items-center gap-1.5 ${
                scope.size >= 2 ? "cursor-pointer" : "cursor-not-allowed opacity-50"
              }`}
              title={scope.size >= 2 ? "Retrieve from each selected document and compare" : "Select two or more documents to compare"}
            >
              <input
                type="checkbox"
                checked={compareOn}
                disabled={scope.size < 2}
                onChange={(e) => setCompare(e.target.checked)}
              />
              Compare
            </label>
          </div>
        )}

        <div className="flex-1 space-y-5 overflow-auto p-4">
          {messages.length === 0 && (
            <p className="mt-10 text-center text-zinc-500">
              {docs.length === 0
                ? "No ready documents yet. Upload one on the Documents page."
                : "Ask a question about your documents."}
            </p>
          )}
          {messages.map((m) => (
            <ChatMessage
              key={m.id}
              msg={m}
              onCite={openCitation}
              onClosest={openClosest}
              activeChunk={viewer?.citation.chunk_id}
            />
          ))}
          <div ref={bottom} />
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            void send();
          }}
          className="flex gap-2 border-t border-zinc-200 p-3 dark:border-zinc-800"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your documents…"
            className="flex-1 rounded border border-zinc-300 bg-transparent px-3 py-2 dark:border-zinc-700"
          />
          {busy ? (
            <button
              type="button"
              onClick={() => abort.current?.abort()}
              className="rounded border border-zinc-300 px-4 py-2 dark:border-zinc-700"
            >
              Stop
            </button>
          ) : (
            <button
              disabled={!input.trim()}
              className="rounded bg-zinc-900 px-4 py-2 text-white disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900"
            >
              Send
            </button>
          )}
        </form>
      </section>

      {viewer && (
        <aside className="hidden w-[34rem] shrink-0 border-l border-zinc-200 lg:block dark:border-zinc-800">
          <div className="flex h-full flex-col">
            <button
              onClick={() => setViewer(null)}
              className="self-end px-3 pt-1 text-xs text-zinc-500 underline"
            >
              Close
            </button>
            <div className="min-h-0 flex-1">
              <PdfPanel
                key={`${viewer.citation.document_id}:${viewer.nonce}`}
                documentId={viewer.citation.document_id}
                initialPage={viewer.citation.page_start}
                highlights={viewer.citation.bbox_list}
                title={viewer.citation.document_title}
              />
            </div>
          </div>
        </aside>
      )}
    </div>
  );
}
