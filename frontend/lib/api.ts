export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const USER_KEY = "rp_user";

export type Session = { userId: string; email: string };

// --- session storage exposed as an external store so React can subscribe safely -------------
const listeners = new Set<() => void>();
let cached: { raw: string | null; value: Session | null } = { raw: null, value: null };

export function subscribeSession(cb: () => void): () => void {
  listeners.add(cb);
  window.addEventListener("storage", cb);
  return () => {
    listeners.delete(cb);
    window.removeEventListener("storage", cb);
  };
}

export function getSessionSnapshot(): Session | null {
  const raw = localStorage.getItem(USER_KEY);
  if (raw !== cached.raw) {
    let value: Session | null = null;
    try {
      value = raw ? (JSON.parse(raw) as Session) : null;
    } catch {
      value = null;
    }
    cached = { raw, value };
  }
  return cached.value;
}

export const getServerSession = (): Session | null => null;

function setSession(s: Session | null) {
  if (s) localStorage.setItem(USER_KEY, JSON.stringify(s));
  else localStorage.removeItem(USER_KEY);
  listeners.forEach((l) => l());
}

export const logout = () => setSession(null);

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const s = getSessionSnapshot();
  return { ...(s ? { "X-User-Id": s.userId } : {}), ...extra };
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : detail;
    } catch {}
    throw new Error(detail);
  }
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
export async function login(email: string): Promise<void> {
  const res = await fetch(`${API_URL}/auth/dev-user`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email }),
  });
  const { user_id } = await json<{ user_id: string }>(res);
  setSession({ userId: user_id, email });
}

export async function listDocuments(): Promise<Doc[]> {
  return json(await fetch(`${API_URL}/documents`, { headers: headers() }));
}

export async function addUrl(url: string): Promise<Doc> {
  return json(
    await fetch(`${API_URL}/documents/url`, {
      method: "POST",
      headers: headers({ "content-type": "application/json" }),
      body: JSON.stringify({ url }),
    }),
  );
}

/** XHR rather than fetch because fetch cannot report upload progress. */
export function uploadDocument(file: File, onProgress: (fraction: number) => void): Promise<Doc> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/documents`);
    for (const [k, v] of Object.entries(headers())) xhr.setRequestHeader(k, v);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
      else {
        let detail = xhr.statusText;
        try {
          detail = JSON.parse(xhr.responseText).detail ?? detail;
        } catch {}
        reject(new Error(detail));
      }
    };
    xhr.onerror = () => reject(new Error("Network error"));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

export async function fetchPdf(documentId: string): Promise<Blob> {
  const res = await fetch(`${API_URL}/documents/${documentId}/file`, { headers: headers() });
  if (!res.ok) throw new Error("Could not load the PDF");
  return res.blob();
}

export type ChatEvent =
  | { type: "status"; stage: string }
  | { type: "token"; text: string }
  | { type: "citation"; citation: Citation }
  | { type: "done"; answer: string; conversationId: string; messageId: string }
  | { type: "error"; message: string };

/** POST + SSE over fetch (EventSource cannot send a body or custom headers). */
export async function streamChat(
  body: { question: string; conversation_id?: string; document_ids?: string[] },
  onEvent: (e: ChatEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_URL}/chat/stream`, {
    method: "POST",
    headers: headers({ "content-type": "application/json" }),
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {}
    onEvent({ type: "error", message: detail });
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
    case "done":
      return { type: "done", answer: d.answer, conversationId: d.conversation_id, messageId: d.message_id };
    case "error":
      return { type: "error", message: d.message };
    default:
      return null;
  }
}
