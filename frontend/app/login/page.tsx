"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { ApiError, login, register } from "@/lib/session";
import { useSession } from "@/lib/useSession";

const MIN_PASSWORD = 10;

export default function LoginPage() {
  const router = useRouter();
  const session = useSession(false);
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // A valid refresh cookie means the visitor is already signed in.
  useEffect(() => {
    if (session.status === "authed") router.replace("/documents");
  }, [session.status, router]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (mode === "signup" && password.length < MIN_PASSWORD) {
      setError(`Use at least ${MIN_PASSWORD} characters for your password.`);
      return;
    }
    setBusy(true);
    try {
      await (mode === "signin" ? login : register)(email.trim(), password);
      router.replace("/documents");
    } catch (err) {
      const e = err as ApiError;
      setError(e.retryAfter ? `${e.message} Try again in ${Math.ceil(e.retryAfter / 60)} min.` : e.message || "Could not sign in");
    } finally {
      setBusy(false);
    }
  }

  if (session.status === "loading") return <p className="p-8 text-sm text-zinc-500">Loading…</p>;
  if (session.status === "offline")
    return <p className="p-8 text-sm text-red-600">Can&apos;t reach the server. Check that the API is running, then reload.</p>;

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <form onSubmit={submit} className="w-full max-w-sm space-y-4">
        <h1 className="text-2xl font-semibold">ResearchPilot</h1>
        <div role="tablist" className="flex gap-1 border-b border-zinc-200 dark:border-zinc-800">
          {(["signin", "signup"] as const).map((m) => (
            <button
              key={m}
              type="button"
              role="tab"
              aria-selected={mode === m}
              onClick={() => {
                setMode(m);
                setError(null);
              }}
              className={`px-3 py-1.5 text-sm ${mode === m ? "border-b-2 border-zinc-900 font-medium dark:border-zinc-100" : "text-zinc-500"}`}
            >
              {m === "signin" ? "Sign in" : "Create account"}
            </button>
          ))}
        </div>
        <label className="block text-sm">
          <span className="mb-1 block text-zinc-500">Email</span>
          <input
            type="email"
            required
            autoFocus
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded border border-zinc-300 bg-transparent px-3 py-2 dark:border-zinc-700"
          />
        </label>
        <label className="block text-sm">
          <span className="mb-1 block text-zinc-500">Password</span>
          <input
            type="password"
            required
            autoComplete={mode === "signin" ? "current-password" : "new-password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded border border-zinc-300 bg-transparent px-3 py-2 dark:border-zinc-700"
          />
          {mode === "signup" && <span className="mt-1 block text-xs text-zinc-500">At least {MIN_PASSWORD} characters.</span>}
        </label>
        {error && (
          <p role="alert" className="text-sm text-red-600">
            {error}
          </p>
        )}
        <button
          disabled={busy || !email || !password}
          className="w-full rounded bg-zinc-900 px-3 py-2 text-white disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900"
        >
          {busy ? "Please wait…" : mode === "signin" ? "Sign in" : "Create account"}
        </button>
      </form>
    </div>
  );
}
