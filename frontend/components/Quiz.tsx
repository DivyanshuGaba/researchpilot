"use client";

import { useState } from "react";
import type { BBox, Quiz as QuizData } from "@/lib/api";
import { scoreQuiz } from "@/lib/study";

type Props = { quiz: QuizData; onSource: (page: number, highlights: BBox[]) => void };

export default function Quiz({ quiz, onSource }: Props) {
  const [answers, setAnswers] = useState<(number | null)[]>(() => quiz.questions.map(() => null));
  const { correct, answered } = scoreQuiz(quiz.questions, answers);
  const done = answered === quiz.questions.length;

  return (
    <div className="space-y-5">
      {quiz.questions.map((q, i) => {
        const picked = answers[i];
        return (
          <fieldset key={i} className="rounded-lg border border-zinc-200 p-4 dark:border-zinc-800">
            <legend className="px-1 text-sm text-zinc-500">Question {i + 1}</legend>
            <p className="mb-3 font-medium">{q.question}</p>
            <div className="space-y-2">
              {q.options.map((opt, j) => {
                const revealed = picked !== null;
                const isRight = j === q.answer_index;
                const style = !revealed
                  ? "border-zinc-300 hover:bg-zinc-50 dark:border-zinc-700 dark:hover:bg-zinc-900"
                  : isRight
                    ? "border-green-500 bg-green-50 dark:bg-green-950"
                    : j === picked
                      ? "border-red-500 bg-red-50 dark:bg-red-950"
                      : "border-zinc-200 opacity-60 dark:border-zinc-800";
                return (
                  <button
                    key={j}
                    disabled={revealed}
                    onClick={() => setAnswers((a) => a.map((x, k) => (k === i ? j : x)))}
                    className={`block w-full rounded border px-3 py-2 text-left text-sm ${style}`}
                  >
                    <span className="mr-2 font-mono text-xs text-zinc-500">{"ABCD"[j]}</span>
                    {opt}
                  </button>
                );
              })}
            </div>
            {picked !== null && (
              <div className="mt-3 text-sm">
                <p className={picked === q.answer_index ? "text-green-700 dark:text-green-400" : "text-red-700 dark:text-red-400"}>
                  {picked === q.answer_index ? "Correct." : `Not quite. The answer is ${"ABCD"[q.answer_index]}.`}
                </p>
                <p className="mt-1 text-zinc-600 dark:text-zinc-400">{q.explanation}</p>
                <button className="mt-1 text-xs underline" onClick={() => onSource(q.page_start, q.bbox_list)}>
                  Source: p.{q.page_start}
                  {q.page_end !== q.page_start && `–${q.page_end}`}
                </button>
              </div>
            )}
          </fieldset>
        );
      })}
      <p className="text-sm text-zinc-500" aria-live="polite">
        {done
          ? `Finished: ${correct} of ${quiz.questions.length} correct.`
          : `${answered} of ${quiz.questions.length} answered${answered ? `, ${correct} correct so far` : ""}.`}
      </p>
      {quiz.questions.length < quiz.requested && (
        <p className="text-xs text-zinc-500">
          The model returned {quiz.questions.length} of the {quiz.requested} questions requested.
        </p>
      )}
    </div>
  );
}
