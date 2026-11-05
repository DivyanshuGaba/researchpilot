"use client";

import { useRouter } from "next/navigation";
import { useEffect, useSyncExternalStore } from "react";
import { bootstrapSession, getServerSnapshot, getSnapshot, subscribe, type SessionState } from "./session";

/**
 * The current session. Restores it from the refresh cookie on first use, and sends anonymous
 * visitors to /login once the check has finished (never before: "loading" is not "signed out").
 */
export function useSession(redirectIfAnonymous = true): SessionState {
  const session = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const router = useRouter();
  useEffect(() => {
    if (session.status === "loading") void bootstrapSession();
  }, [session.status]);
  useEffect(() => {
    if (redirectIfAnonymous && session.status === "anon") router.replace("/login");
  }, [redirectIfAnonymous, session.status, router]);
  return session;
}
