import { describe, expect, it } from "vitest";
import { parseFrame } from "../api";

describe("parseFrame", () => {
  it("parses a token event", () => {
    expect(parseFrame('event: token\ndata: {"text": "hi "}')).toEqual({ type: "token", text: "hi " });
  });

  it("ignores heartbeat comments", () => {
    expect(parseFrame(": ping")).toBeNull();
  });

  it("maps done and error events", () => {
    expect(
      parseFrame('event: done\ndata: {"answer": "A", "conversation_id": "c1", "message_id": "m1"}'),
    ).toEqual({
      type: "done",
      answer: "A",
      conversationId: "c1",
      messageId: "m1",
      standaloneQuery: "",
      closest: [],
      degraded: [],
    });
    expect(parseFrame('event: error\ndata: {"message": "boom"}')).toEqual({ type: "error", message: "boom" });
  });

  it("parses closest passages and the rewritten query", () => {
    const passage = { document_id: "d", document_title: "T", page_start: 3, quote: "q", bbox_list: [], score: 0.02 };
    expect(parseFrame(`event: closest\ndata: ${JSON.stringify({ passages: [passage] })}`)).toEqual({
      type: "closest",
      passages: [passage],
    });
    const done = parseFrame(
      'event: done\ndata: {"answer": "A", "conversation_id": "c", "message_id": "m", "standalone_query": "what is X", "degraded": ["rerank"]}',
    );
    expect(done).toMatchObject({ standaloneQuery: "what is X", degraded: ["rerank"] });
  });

  it("returns null for unknown or empty frames", () => {
    expect(parseFrame("event: mystery\ndata: {}")).toBeNull();
    expect(parseFrame("")).toBeNull();
  });
});
