import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  authFetch,
  bootstrapSession,
  errorFrom,
  getSnapshot,
  login,
  logout,
  resetSessionForTests,
} from "../session";

const tokenBody = (n = 1, extra = {}) => ({
  access_token: `tok-${n}`,
  expires_in: 900,
  email: "a@example.com",
  user_id: "u1",
  is_admin: false,
  ...extra,
});

const ok = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  resetSessionForTests();
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

const calls = (path: string) => fetchMock.mock.calls.filter((c) => String(c[0]).endsWith(path));

describe("session bootstrap", () => {
  it("restores a session from the refresh cookie", async () => {
    fetchMock.mockResolvedValueOnce(ok(tokenBody(1, { is_admin: true })));
    await bootstrapSession();
    expect(getSnapshot()).toMatchObject({ status: "authed", email: "a@example.com", isAdmin: true });
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: "POST", credentials: "include" });
  });

  it("becomes anonymous when there is no valid cookie", async () => {
    fetchMock.mockResolvedValueOnce(ok({ detail: "not signed in" }, 401));
    await bootstrapSession();
    expect(getSnapshot().status).toBe("anon");
  });

  it("reports offline, not signed out, when the server is unreachable", async () => {
    fetchMock.mockRejectedValueOnce(new TypeError("network"));
    await bootstrapSession();
    expect(getSnapshot().status).toBe("offline");
    fetchMock.mockResolvedValueOnce(ok(tokenBody()));
    await bootstrapSession(); // a later attempt can still succeed
    expect(getSnapshot().status).toBe("authed");
  });
});

describe("authFetch", () => {
  it("sends the bearer token and always includes credentials", async () => {
    fetchMock.mockResolvedValueOnce(ok(tokenBody(1))); // refresh
    fetchMock.mockResolvedValueOnce(ok({ fine: true }));
    const res = await authFetch("/documents");
    expect(res.status).toBe(200);
    const [, init] = fetchMock.mock.calls[1];
    expect(init.headers.Authorization).toBe("Bearer tok-1");
    expect(init.credentials).toBe("include");
  });

  it("reuses the token instead of refreshing on every request", async () => {
    fetchMock.mockResolvedValueOnce(ok(tokenBody(1)));
    fetchMock.mockResolvedValue(ok({}));
    await authFetch("/a");
    await authFetch("/b");
    expect(calls("/auth/refresh")).toHaveLength(1);
  });

  it("refreshes once and retries when a request returns 401", async () => {
    fetchMock.mockResolvedValueOnce(ok(tokenBody(1))); // initial refresh
    fetchMock.mockResolvedValueOnce(ok({ detail: "expired" }, 401)); // first call
    fetchMock.mockResolvedValueOnce(ok(tokenBody(2))); // refresh
    fetchMock.mockResolvedValueOnce(ok({ done: true })); // retry
    const res = await authFetch("/documents");
    expect(res.status).toBe(200);
    expect(fetchMock.mock.calls[3][1].headers.Authorization).toBe("Bearer tok-2");
    expect(calls("/documents")).toHaveLength(2);
  });

  it("signs the user out when the refresh fails after a 401", async () => {
    fetchMock.mockResolvedValueOnce(ok(tokenBody(1)));
    fetchMock.mockResolvedValueOnce(ok({ detail: "expired" }, 401));
    fetchMock.mockResolvedValueOnce(ok({ detail: "session expired" }, 401));
    const res = await authFetch("/documents");
    expect(res.status).toBe(401);
    expect(getSnapshot().status).toBe("anon");
  });

  it("shares one refresh between concurrent requests", async () => {
    let release: (r: Response) => void = () => {};
    fetchMock.mockImplementationOnce(() => new Promise<Response>((r) => (release = r)));
    fetchMock.mockResolvedValue(ok({}));
    const both = Promise.all([authFetch("/a"), authFetch("/b")]);
    await Promise.resolve();
    release(ok(tokenBody(1)));
    await both;
    expect(calls("/auth/refresh")).toHaveLength(1);
  });

  it("throws a clear error when there is no session at all", async () => {
    fetchMock.mockResolvedValueOnce(ok({ detail: "not signed in" }, 401));
    await expect(authFetch("/documents")).rejects.toMatchObject({ status: 401 });
  });
});

describe("login and logout", () => {
  it("stores the session on success and shows the server's message on failure", async () => {
    fetchMock.mockResolvedValueOnce(ok(tokenBody()));
    await login("a@example.com", "correct horse battery");
    expect(getSnapshot().status).toBe("authed");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ email: "a@example.com", password: "correct horse battery" });

    resetSessionForTests();
    fetchMock.mockResolvedValueOnce(ok({ detail: "Invalid email or password" }, 401));
    await expect(login("a@example.com", "bad")).rejects.toThrow("Invalid email or password");
    expect(getSnapshot().status).not.toBe("authed");
  });

  it("clears the session even if the logout request fails", async () => {
    fetchMock.mockResolvedValueOnce(ok(tokenBody()));
    await login("a@example.com", "pw-long-enough");
    fetchMock.mockRejectedValueOnce(new TypeError("network"));
    await expect(logout()).rejects.toThrow();
    expect(getSnapshot().status).toBe("anon");
  });

  it("never writes credentials to browser storage", async () => {
    const store = vi.fn();
    vi.stubGlobal("localStorage", { setItem: store, getItem: vi.fn(), removeItem: vi.fn() });
    fetchMock.mockResolvedValueOnce(ok(tokenBody()));
    await login("a@example.com", "pw-long-enough");
    expect(store).not.toHaveBeenCalled();
  });
});

describe("errorFrom", () => {
  it("reads string, object and validation-style details, and Retry-After", async () => {
    const a = await errorFrom(new Response(JSON.stringify({ detail: "too many requests" }), { status: 429, headers: { "retry-after": "12" } }));
    expect(a).toBeInstanceOf(ApiError);
    expect(a).toMatchObject({ status: 429, message: "too many requests", retryAfter: 12 });

    const b = await errorFrom(new Response(JSON.stringify({ detail: { message: "model busy", closest: [{ x: 1 }] } }), { status: 503 }));
    expect(b.message).toBe("model busy");
    expect((b.detail as { closest: unknown[] }).closest).toHaveLength(1);

    const c = await errorFrom(new Response(JSON.stringify({ detail: [{ msg: "value is not a valid email" }] }), { status: 422 }));
    expect(c.message).toBe("value is not a valid email");

    const d = await errorFrom(new Response("<html>bad gateway</html>", { status: 502, statusText: "Bad Gateway" }));
    expect(d.message).toBe("Bad Gateway");
  });
});
