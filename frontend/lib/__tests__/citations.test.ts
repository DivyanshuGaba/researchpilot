import { describe, expect, it } from "vitest";
import type { Citation } from "../api";
import { segmentAnswer } from "../citations";

const cite = (marker: string, page: number): Citation => ({
  marker,
  chunk_id: marker,
  document_id: "d",
  document_title: "Doc",
  page_start: page,
  page_end: page,
  quote: "",
  bbox_list: [],
});

describe("segmentAnswer", () => {
  it("turns known markers into citation segments", () => {
    const segs = segmentAnswer("ATP [c_1] is made.", [cite("c_1", 7)], false);
    expect(segs.map((s) => s.kind)).toEqual(["text", "cite", "text"]);
  });

  it("shows unresolved markers as pending while streaming", () => {
    expect(segmentAnswer("a [c_2]", [], true).map((s) => s.kind)).toEqual(["text", "pending"]);
  });

  it("drops unknown markers once the answer is final", () => {
    expect(segmentAnswer("a [c_9] b", [], false)).toEqual([
      { kind: "text", text: "a " },
      { kind: "text", text: " b" },
    ]);
  });
});
