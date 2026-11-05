import type { QuizQuestion } from "./api";

export type PageRefPart = { kind: "text"; text: string } | { kind: "page"; page: number; label: string };

const PAGE_REF = /[(\[]p\.(\d+)(?:\s*[-–]\s*(\d+))?[)\]]/g;

/** Split summary text so "(p.4)" / "[p.3-5]" references can become clickable. */
export function splitPageRefs(text: string): PageRefPart[] {
  const out: PageRefPart[] = [];
  let last = 0;
  for (const m of text.matchAll(PAGE_REF)) {
    const idx = m.index ?? 0;
    if (idx > last) out.push({ kind: "text", text: text.slice(last, idx) });
    out.push({ kind: "page", page: Number(m[1]), label: m[0] });
    last = idx + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

export type Block = { kind: "p"; text: string } | { kind: "ul"; items: string[] };

/** Tiny block parser for model summaries: paragraphs and "- " / "* " bullet lists. */
export function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  let list: string[] | null = null;
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    const bullet = /^[-*•]\s+(.*)$/.exec(line);
    if (bullet) {
      if (!list) {
        list = [];
        blocks.push({ kind: "ul", items: list });
      }
      list.push(bullet[1]);
    } else {
      list = null;
      if (line) blocks.push({ kind: "p", text: line });
    }
  }
  return blocks;
}

export function scoreQuiz(questions: QuizQuestion[], answers: (number | null)[]): { correct: number; answered: number } {
  let correct = 0;
  let answered = 0;
  questions.forEach((q, i) => {
    const a = answers[i];
    if (a === null || a === undefined) return;
    answered++;
    if (a === q.answer_index) correct++;
  });
  return { correct, answered };
}
