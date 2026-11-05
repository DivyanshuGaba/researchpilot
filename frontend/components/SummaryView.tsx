"use client";

import type { Summary } from "@/lib/api";
import { parseBlocks, splitPageRefs } from "@/lib/study";

function Inline({ text, onPage }: { text: string; onPage: (page: number) => void }) {
  return (
    <>
      {splitPageRefs(text).map((p, i) =>
        p.kind === "text" ? (
          <span key={i}>{p.text}</span>
        ) : (
          <button
            key={i}
            onClick={() => onPage(p.page)}
            className="mx-0.5 rounded-full border border-zinc-300 px-1.5 text-xs hover:bg-yellow-100 dark:border-zinc-700 dark:hover:bg-yellow-950"
          >
            {p.label.replace(/[()[\]]/g, "")}
          </button>
        ),
      )}
    </>
  );
}

export default function SummaryView({ summary, onPage }: { summary: Summary; onPage: (page: number) => void }) {
  return (
    <div className="space-y-3 leading-relaxed">
      {parseBlocks(summary.summary).map((b, i) =>
        b.kind === "p" ? (
          <p key={i}>
            <Inline text={b.text} onPage={onPage} />
          </p>
        ) : (
          <ul key={i} className="list-disc space-y-1.5 pl-5">
            {b.items.map((item, j) => (
              <li key={j}>
                <Inline text={item} onPage={onPage} />
              </li>
            ))}
          </ul>
        ),
      )}
      <p className="text-xs text-zinc-500">
        {summary.cached ? "Cached summary" : "Freshly generated"} from {summary.sections} section
        {summary.sections === 1 ? "" : "s"}.
        {summary.truncated && " The document is long, so sections were sampled evenly."}
      </p>
    </div>
  );
}
