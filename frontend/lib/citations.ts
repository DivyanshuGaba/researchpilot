import type { Citation } from "./api";

export type Segment =
  | { kind: "text"; text: string }
  | { kind: "cite"; citation: Citation }
  | { kind: "pending"; marker: string };

const MARKER = /\[(c_\d+)\]/g;

/**
 * Split an answer into text and citation segments. While streaming, a marker whose citation has
 * not arrived yet is "pending"; once the answer is final, unknown markers are dropped.
 */
export function segmentAnswer(text: string, citations: Citation[], streaming: boolean): Segment[] {
  const byMarker = new Map(citations.map((c) => [c.marker, c]));
  const out: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(MARKER)) {
    const idx = m.index ?? 0;
    if (idx > last) out.push({ kind: "text", text: text.slice(last, idx) });
    const c = byMarker.get(m[1]);
    if (c) out.push({ kind: "cite", citation: c });
    else if (streaming) out.push({ kind: "pending", marker: m[1] });
    last = idx + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}
