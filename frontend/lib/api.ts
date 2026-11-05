import { API_URL, ApiError, authFetch, errorFrom, getToken } from "./session";

export { API_URL, ApiError };

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw await errorFrom(res);
  return res.json() as Promise<T>;
}

// --- types ---------------------------------------------------------------------------------
export type Doc = {
  id: string;
  title: string;
  source_type: "pdf" | "docx" | "url";
  status: "queued" | "processing" | "ready" | "failed";
  stage: string;
  page_count: number;
};

export type BBox = { page: number; bbox: [number, number, number, number] };

export type Citation = {
  marker: string;
  chunk_id: string;
  document_id: string;
  document_title: string;
  page_start: number;
  page_end: number;
  quote: string;
  bbox_list: BBox[];
};

// --- calls ---------------------------------------------------------------------------------
export async function listDocuments(): Promise<Doc[]> {
  return json(await authFetch("/documents"));
}

export async function addUrl(url: string): Promise<Doc> {
  return json(
    await authFetch("/documents/url", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ url }),
    }),
  );
}

/** XHR rather than fetch because fetch cannot report upload progress. */
export async function uploadDocument(file: File, onProgress: (fraction: number) => void): Promise<Doc> {
  const token = await getToken();
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/documents`);
    xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
      else {
        let detail = xhr.statusText;
        try {
          const d = JSON.parse(xhr.responseText).detail;
          detail = typeof d === "string" ? d : (d?.message ?? detail);
        } catch {}
        reject(new ApiError(xhr.status, detail));
      }
    };
    xhr.onerror = () => reject(new Error("Network error"));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

export async function fetchPdf(documentId: string): Promise<Blob> {
  const res = await authFetch(`/documents/${documentId}/file`);
  if (!res.ok) throw new Error("Could not load the PDF");
  return res.blob();
}

/** A passage shown when the documents do not cover the question (the confidence gate fired). */
export type Closest = {
  document_id: string;
  document_title: string;
  page_start: number;
  quote: string;
  bbox_list: BBox[];
  score: number;
};

export type ToolInfo = {
  id: string;
  name: string;
  input: Record<string, unknown>;
  status: "started" | "finished";
  duration_ms?: number;
  is_error?: boolean;
};

export type ChatEvent =
  | { type: "status"; stage: string }
  | { type: "tool"; tool: ToolInfo }
  | { type: "token"; text: string }
  | { type: "citation"; citation: Citation }
  | { type: "closest"; passages: Closest[] }
  | {
      type: "done";
      answer: string;
      conversationId: string;
      messageId: string;
      standaloneQuery: string;
      closest: Closest[];
      degraded: string[];
      stopped: string | null;
      cacheStatus: string;
    }
  | { type: "error"; message: string; code?: string; retryAfter?: number };

/** POST + SSE over fetch (EventSource cannot send a body or custom headers). */
export async function streamChat(
  body: {
    question: string;
    conversation_id?: string;
    document_ids?: string[];
    mode?: "qa" | "compare" | "agent";
  },
  onEvent: (e: ChatEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const res = await authFetch("/chat/stream", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    const err = await errorFrom(res);
    onEvent({ type: "error", message: err.message, code: String(err.status), retryAfter: err.retryAfter });
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const ev = parseFrame(frame);
      if (ev) onEvent(ev);
    }
  }
}

export function parseFrame(frame: string): ChatEvent | null {
  let name = "";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith(":")) return null; // heartbeat comment
    if (line.startsWith("event: ")) name = line.slice(7);
    else if (line.startsWith("data: ")) data += line.slice(6);
  }
  if (!name || !data) return null;
  const d = JSON.parse(data);
  switch (name) {
    case "status":
      return { type: "status", stage: d.stage };
    case "token":
      return { type: "token", text: d.text };
    case "citation":
      return { type: "citation", citation: d as Citation };
    case "tool":
      return { type: "tool", tool: d as ToolInfo };
    case "closest":
      return { type: "closest", passages: d.passages as Closest[] };
    case "done":
      return {
        type: "done",
        answer: d.answer,
        conversationId: d.conversation_id,
        messageId: d.message_id,
        standaloneQuery: d.standalone_query ?? "",
        closest: (d.closest ?? []) as Closest[],
        degraded: d.degraded ?? [],
        stopped: d.stopped ?? null,
        cacheStatus: d.cache_status ?? "miss",
      };
    case "error":
      return { type: "error", message: d.message, code: d.code, retryAfter: d.retry_after ?? undefined };
    default:
      return null;
  }
}

// --- study ---------------------------------------------------------------------------------

export type QuizQuestion = {
  question: string;
  options: string[];
  answer_index: number;
  explanation: string;
  chunk_id: string;
  page_start: number;
  page_end: number;
  bbox_list: BBox[];
};
export type Quiz = { id: string; document_id: string; requested: number; questions: QuizQuestion[] };

export type Card = {
  front: string;
  back: string;
  chunk_id: string;
  page_start: number;
  page_end: number;
  bbox_list: BBox[];
};
export type CardSet = { id: string; document_id: string; requested: number; cards: Card[] };

export type Summary = {
  document_id: string;
  summary: string;
  cached: boolean;
  sections: number;
  truncated: boolean;
};

async function post<T>(path: string, body: unknown): Promise<T> {
  return json(
    await authFetch(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

export const createQuiz = (document_id: string, num_questions: number) =>
  post<Quiz>("/study/quizzes", { document_id, num_questions });
export const createFlashcards = (document_id: string, num_cards: number) =>
  post<CardSet>("/study/flashcards", { document_id, num_cards });
export const createSummary = (document_id: string) => post<Summary>("/study/summaries", { document_id });

// --- admin ---------------------------------------------------------------------------------

export type Metrics = {
  window_days: number;
  queries: number;
  active_users: number;
  latency_ms: { p50: number | null; p95: number | null };
  stages: Record<string, { p50_ms: number | null; p95_ms: number | null; samples: number }>;
  tokens: { input: number; output: number; cache_read: number };
  cost_usd: { total: number | null; per_query: number | null };
  cache: { by_layer: Record<string, number>; hit_rate: number | null };
  refusal_rate: number | null;
  error_rate: number | null;
  errors_by_type: Record<string, number>;
  modes: Record<string, number>;
  degraded: Record<string, number>;
  daily: { day: string; queries: number; cost_usd: number | null }[];
  ingestion: {
    documents_by_status: Record<string, number>;
    failure_rate: number | null;
    seconds_per_page: number | null;
    queue_depth: number;
  };
};

export async function fetchMetrics(days: number): Promise<Metrics> {
  return json(await authFetch(`/admin/metrics?days=${days}`));
}
