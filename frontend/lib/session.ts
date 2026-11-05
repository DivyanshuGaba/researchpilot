/**
 * Authentication state for the browser.
 *
 * The short-lived access token lives only in memory (never localStorage, so a script injected into
 * the page cannot read a long-lived credential). The refresh token is an HttpOnly cookie the
 * browser sends to /auth/refresh; that is how a reload or a new tab gets back in.
 */

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type SessionState = {
  status: "loading" | "anon" | "authed" | "offline";
  email: string | null;
  userId: string | null;
  isAdmin: boolean;
};

const LOADING: SessionState = { status: "loading", email: null, userId: null, isAdmin: false };
const ANON: SessionState = { status: "anon", email: null, userId: null, isAdmin: false };
const OFFLINE: SessionState = { status: "offline", email: null, userId: null, isAdmin: false };

let state: SessionState = LOADING;
let token: string | null = null;
let expiresAt = 0;
let inflight: Promise<boolean> | null = null;
const listeners = new Set<() => void>();

function setState(next: SessionState) {
  state = next;
  listeners.forEach((l) => l());
}

export function subscribe(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}
export const getSnapshot = (): SessionState => state;
export const getServerSnapshot = (): SessionState => LOADING;

/** Test helper: forget everything. */
export function resetSessionForTests() {
  token = null;
  expiresAt = 0;
  inflight = null;
  state = LOADING;
  listeners.clear();
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public retryAfter?: number,
    public detail?: unknown,
  ) {
    super(message);
  }
}

/** Turn an error response into an ApiError with a message that is safe to show. */
export async function errorFrom(res: Response): Promise<ApiError> {
  let message = res.statusText || `Request failed (${res.status})`;
  let detail: unknown;
  try {
    const body = await res.json();
    detail = body.detail;
    if (typeof detail === "string") message = detail;
    else if (detail && typeof detail === "object" && "message" in detail) message = String((detail as { message: unknown }).message);
    else if (Array.isArray(detail) && detail[0]?.msg) message = String(detail[0].msg);
  } catch {}
  const retry = Number(res.headers.get("retry-after"));
  return new ApiError(res.status, message, Number.isFinite(retry) && retry > 0 ? retry : undefined, detail);
}

type TokenOut = { access_token: string; expires_in: number; email: string; user_id: string; is_admin?: boolean };

function accept(t: TokenOut) {
  token = t.access_token;
  expiresAt = Date.now() + t.expires_in * 1000;
  setState({ status: "authed", email: t.email, userId: t.user_id, isAdmin: !!t.is_admin });
}

function clear() {
  token = null;
  expiresAt = 0;
  setState(ANON);
}

/** Exchange the refresh cookie for a new access token. Concurrent callers share one request. */
export function refreshSession(): Promise<boolean> {
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      const res = await fetch(`${API_URL}/auth/refresh`, { method: "POST", credentials: "include" });
      if (!res.ok) {
        clear();
        return false;
      }
      accept((await res.json()) as TokenOut);
      return true;
    } catch {
      // The server could not be reached: that is not the same as being signed out.
      if (state.status === "loading") setState(OFFLINE);
      return false;
    } finally {
      inflight = null;
    }
  })();
  return inflight;
}

/** Restore a session on page load. */
export async function bootstrapSession(): Promise<void> {
  if (state.status === "loading" || state.status === "offline") await refreshSession();
}

export async function getToken(): Promise<string> {
  if (token && Date.now() < expiresAt - 30_000) return token;
  if ((await refreshSession()) && token) return token;
  throw new ApiError(401, "Please sign in again.");
}

/** fetch with the bearer token; on a 401 it refreshes once and retries. */
export async function authFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const send = async () =>
    fetch(`${API_URL}${path}`, {
      ...init,
      credentials: "include",
      headers: { ...(init.headers as Record<string, string> | undefined), Authorization: `Bearer ${await getToken()}` },
    });
  let res = await send();
  if (res.status === 401) {
    token = null;
    expiresAt = 0;
    if (await refreshSession()) res = await send();
  }
  return res;
}

async function credentialsRequest(path: "/auth/login" | "/auth/register", email: string, password: string) {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    credentials: "include",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) throw await errorFrom(res);
  accept((await res.json()) as TokenOut);
}

export const login = (email: string, password: string) => credentialsRequest("/auth/login", email, password);
export const register = (email: string, password: string) => credentialsRequest("/auth/register", email, password);

export async function logout(): Promise<void> {
  try {
    await fetch(`${API_URL}/auth/logout`, { method: "POST", credentials: "include" });
  } finally {
    clear();
  }
}
