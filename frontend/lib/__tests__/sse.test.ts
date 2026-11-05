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
      stopped: null,
      cacheStatus: "miss",
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

  it("parses tool events", () => {
    const frame =
      'event: tool\ndata: {"id": "t1", "name": "calculator", "input": {"expression": "2+2"}, "status": "finished", "duration_ms": 3, "is_error": false}';
    expect(parseFrame(frame)).toMatchObject({
      type: "tool",
      tool: { id: "t1", name: "calculator", status: "finished", duration_ms: 3 },
    });
  });

  it("parses model-outage errors with their code and retry hint", () => {
    expect(
      parseFrame('event: error\ndata: {"code": "rate_limited", "message": "The model is busy", "retry_after": 30}'),
    ).toEqual({ type: "error", message: "The model is busy", code: "rate_limited", retryAfter: 30 });
    expect(parseFrame('event: error\ndata: {"code": "internal", "message": "oops", "retry_after": null}')).toMatchObject({
      retryAfter: undefined,
    });
  });

  it("reads the cache status from the done event", () => {
    const f = parseFrame('event: done\ndata: {"answer": "A", "conversation_id": "c", "message_id": "m", "cache_status": "semantic"}');
    expect(f).toMatchObject({ type: "done", cacheStatus: "semantic" });
  });

  it("returns null for unknown or empty frames", () => {
    expect(parseFrame("event: mystery\ndata: {}")).toBeNull();
    expect(parseFrame("")).toBeNull();
  });
});
