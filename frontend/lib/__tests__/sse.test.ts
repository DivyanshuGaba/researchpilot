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
    ).toEqual({ type: "done", answer: "A", conversationId: "c1", messageId: "m1" });
    expect(parseFrame('event: error\ndata: {"message": "boom"}')).toEqual({ type: "error", message: "boom" });
  });

  it("returns null for unknown or empty frames", () => {
    expect(parseFrame("event: mystery\ndata: {}")).toBeNull();
    expect(parseFrame("")).toBeNull();
  });
});
