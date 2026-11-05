"use client";

import { useState } from "react";
import type { BBox, CardSet } from "@/lib/api";

type Props = { set: CardSet; onSource: (page: number, highlights: BBox[]) => void };

export default function Flashcards({ set, onSource }: Props) {
  const [index, setIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const card = set.cards[index];
  const go = (delta: number) => {
    setIndex((i) => Math.min(set.cards.length - 1, Math.max(0, i + delta)));
    setFlipped(false);
  };

  return (
    <div className="mx-auto max-w-xl space-y-4">
      <button
        onClick={() => setFlipped((f) => !f)}
        aria-label={flipped ? "Show the question" : "Show the answer"}
        className="flex h-56 w-full items-center justify-center rounded-xl border border-zinc-300 p-6 text-center text-lg shadow-sm transition hover:shadow dark:border-zinc-700"
      >
        <span>
          <span className="mb-2 block text-xs uppercase tracking-wide text-zinc-500">
            {flipped ? "Answer" : "Question"}
          </span>
          {flipped ? card.back : card.front}
        </span>
      </button>
      <div className="flex items-center justify-between text-sm">
        <button
          onClick={() => go(-1)}
          disabled={index === 0}
          className="rounded border border-zinc-300 px-3 py-1 disabled:opacity-40 dark:border-zinc-700"
        >
          ← Previous
        </button>
        <span className="text-zinc-500">
          {index + 1} / {set.cards.length}
        </span>
        <button
          onClick={() => go(1)}
          disabled={index === set.cards.length - 1}
          className="rounded border border-zinc-300 px-3 py-1 disabled:opacity-40 dark:border-zinc-700"
        >
          Next →
        </button>
      </div>
      <p className="text-center text-xs">
        <button className="underline" onClick={() => onSource(card.page_start, card.bbox_list)}>
          Source: p.{card.page_start}
        </button>
      </p>
    </div>
  );
}
