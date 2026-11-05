"use client";

import { useEffect, useState } from "react";
import Flashcards from "@/components/Flashcards";
import PdfPanel from "@/components/PdfPanel";
import Quiz from "@/components/Quiz";
import SummaryView from "@/components/SummaryView";
import {
  createFlashcards,
  createQuiz,
  createSummary,
  listDocuments,
  type BBox,
  type CardSet,
  type Doc,
  type Quiz as QuizData,
  type Summary,
} from "@/lib/api";

type Tab = "quiz" | "cards" | "summary";
const TABS: { id: Tab; label: string }[] = [
  { id: "quiz", label: "Quiz" },
  { id: "cards", label: "Flashcards" },
  { id: "summary", label: "Summary" },
];

export default function StudyPage() {
  const [docs, setDocs] = useState<Doc[] | null>(null);
  const [docId, setDocId] = useState("");
  const [tab, setTab] = useState<Tab>("quiz");
  const [count, setCount] = useState(5);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Results are kept per document so switching tabs or documents does not lose them.
  const [quizzes, setQuizzes] = useState<Record<string, QuizData>>({});
  const [cards, setCards] = useState<Record<string, CardSet>>({});
  const [summaries, setSummaries] = useState<Record<string, Summary>>({});
  const [viewer, setViewer] = useState<{ page: number; highlights: BBox[]; nonce: number } | null>(null);

  useEffect(() => {
    listDocuments()
      .then((d) => {
        const ready = d.filter((x) => x.status === "ready");
        setDocs(ready);
        if (ready.length) setDocId((cur) => cur || ready[0].id);
      })
      .catch((e: Error) => {
        setDocs([]);
        setError(e.message);
      });
  }, []);

  const doc = docs?.find((d) => d.id === docId);
  const openPage = (page: number, highlights: BBox[] = []) =>
    setViewer((v) => ({ page, highlights, nonce: (v?.nonce ?? 0) + 1 }));

  async function generate() {
    if (!docId || busy) return;
    setBusy(true);
    setError(null);
    try {
      if (tab === "quiz") {
        const q = await createQuiz(docId, count);
        setQuizzes((s) => ({ ...s, [docId]: q }));
      } else if (tab === "cards") {
        const c = await createFlashcards(docId, count);
        setCards((s) => ({ ...s, [docId]: c }));
      } else {
        const s = await createSummary(docId);
        setSummaries((m) => ({ ...m, [docId]: s }));
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const verb = tab === "summary" ? "Summarize" : "Generate";
  const hasResult = tab === "quiz" ? quizzes[docId] : tab === "cards" ? cards[docId] : summaries[docId];

  return (
    <div className="flex h-full">
      <section className="min-w-0 flex-1 overflow-auto p-4">
        <div className="mx-auto max-w-2xl space-y-5">
          {docs !== null && docs.length === 0 && (
            <p className="text-zinc-500">No ready documents yet. Upload one on the Documents page first.</p>
          )}
          {docs && docs.length > 0 && (
            <>
              <div className="flex flex-wrap items-end gap-3">
                <label className="text-sm">
                  <span className="mb-1 block text-zinc-500">Document</span>
                  <select
                    value={docId}
                    onChange={(e) => setDocId(e.target.value)}
                    className="rounded border border-zinc-300 bg-transparent px-2 py-1.5 dark:border-zinc-700"
                  >
                    {docs.map((d) => (
                      <option key={d.id} value={d.id} className="text-black">
                        {d.title}
                      </option>
                    ))}
                  </select>
                </label>
                {tab !== "summary" && (
                  <label className="text-sm">
                    <span className="mb-1 block text-zinc-500">{tab === "quiz" ? "Questions" : "Cards"}</span>
                    <select
                      value={count}
                      onChange={(e) => setCount(Number(e.target.value))}
                      className="rounded border border-zinc-300 bg-transparent px-2 py-1.5 dark:border-zinc-700"
                    >
                      {[3, 5, 10, 15, 20].map((n) => (
                        <option key={n} value={n} className="text-black">
                          {n}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                <button
                  onClick={() => void generate()}
                  disabled={busy || !docId}
                  className="rounded bg-zinc-900 px-4 py-1.5 text-white disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900"
                >
                  {busy ? "Working…" : hasResult ? `${verb} again` : verb}
                </button>
              </div>

              <div role="tablist" className="flex gap-1 border-b border-zinc-200 dark:border-zinc-800">
                {TABS.map((t) => (
                  <button
                    key={t.id}
                    role="tab"
                    aria-selected={tab === t.id}
                    onClick={() => setTab(t.id)}
                    className={`px-3 py-1.5 text-sm ${
                      tab === t.id ? "border-b-2 border-zinc-900 font-medium dark:border-zinc-100" : "text-zinc-500"
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>

              {error && <p className="text-sm text-red-600">{error}</p>}
              {busy && (
                <p className="text-sm text-zinc-500">
                  {tab === "summary"
                    ? "Summarizing section by section, then combining. Long documents take a little while…"
                    : "Reading the document and writing questions…"}
                </p>
              )}

              {tab === "quiz" && quizzes[docId] && (
                <Quiz key={quizzes[docId].id} quiz={quizzes[docId]} onSource={openPage} />
              )}
              {tab === "cards" && cards[docId] && (
                <Flashcards key={cards[docId].id} set={cards[docId]} onSource={openPage} />
              )}
              {tab === "summary" && summaries[docId] && (
                <SummaryView summary={summaries[docId]} onPage={(p) => openPage(p)} />
              )}
              {!hasResult && !busy && (
                <p className="text-sm text-zinc-500">
                  {tab === "summary"
                    ? "Create a summary with page references you can click."
                    : `Generate ${tab === "quiz" ? "a quiz" : "flashcards"} from this document.`}
                </p>
              )}
            </>
          )}
        </div>
      </section>

      {viewer && doc && (
        <aside className="hidden w-[34rem] shrink-0 border-l border-zinc-200 lg:block dark:border-zinc-800">
          <div className="flex h-full flex-col">
            <button onClick={() => setViewer(null)} className="self-end px-3 pt-1 text-xs text-zinc-500 underline">
              Close
            </button>
            <div className="min-h-0 flex-1">
              <PdfPanel
                key={`${doc.id}:${viewer.nonce}`}
                documentId={doc.id}
                initialPage={viewer.page}
                highlights={viewer.highlights}
                title={doc.title}
              />
            </div>
          </div>
        </aside>
      )}
    </div>
  );
}
