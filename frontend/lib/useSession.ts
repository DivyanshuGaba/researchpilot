"use client";

import { useRouter } from "next/navigation";
import { useEffect, useSyncExternalStore } from "react";
import { getServerSession, getSessionSnapshot, subscribeSession, type Session } from "./api";

const noopSubscribe = () => () => {};

/** False during server render and hydration, true once the browser snapshot is readable. */
export function useHydrated(): boolean {
  return useSyncExternalStore(noopSubscribe, () => true, () => false);
}

/**
 * Returns the signed-in session. Redirects to /login only after hydration: before that the
 * session is unknown (server snapshot is null), so redirecting would sign people out on reload.
 */
export function useSession(): Session | null {
  const session = useSyncExternalStore(subscribeSession, getSessionSnapshot, getServerSession);
  const hydrated = useHydrated();
  const router = useRouter();
  useEffect(() => {
    if (hydrated && !session) router.replace("/login");
  }, [hydrated, session, router]);
  return session;
}
