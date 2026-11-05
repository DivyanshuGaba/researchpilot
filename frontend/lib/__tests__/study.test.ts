import { describe, expect, it } from "vitest";
import type { QuizQuestion } from "../api";
import { parseBlocks, scoreQuiz, splitPageRefs } from "../study";

describe("splitPageRefs", () => {
  it("finds single pages and ranges in parentheses or brackets", () => {
    const parts = splitPageRefs("Key finding (p.4). Also [p.3-5] matters.");
    expect(parts.filter((p) => p.kind === "page")).toEqual([
      { kind: "page", page: 4, label: "(p.4)" },
      { kind: "page", page: 3, label: "[p.3-5]" },
    ]);
    expect(parts.map((p) => (p.kind === "text" ? p.text : p.label)).join("")).toBe(
      "Key finding (p.4). Also [p.3-5] matters.",
    );
  });

  it("leaves text without references alone", () => {
    expect(splitPageRefs("nothing here")).toEqual([{ kind: "text", text: "nothing here" }]);
  });
});

describe("parseBlocks", () => {
  it("groups bullets into lists and keeps paragraphs", () => {
    expect(parseBlocks("Overview.\n- one (p.1)\n- two\n\nClosing.")).toEqual([
      { kind: "p", text: "Overview." },
      { kind: "ul", items: ["one (p.1)", "two"] },
      { kind: "p", text: "Closing." },
    ]);
  });
});

describe("scoreQuiz", () => {
  const q = (answer_index: number): QuizQuestion => ({
    question: "q", options: ["a", "b", "c", "d"], answer_index, explanation: "",
    chunk_id: "c", page_start: 1, page_end: 1, bbox_list: [],
  });
  it("counts only answered questions", () => {
    expect(scoreQuiz([q(0), q(1), q(2)], [0, 3, null])).toEqual({ correct: 1, answered: 2 });
    expect(scoreQuiz([q(0)], [])).toEqual({ correct: 0, answered: 0 });
  });
});
