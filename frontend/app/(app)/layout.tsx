"use client";

import Nav from "@/components/Nav";
import { useSession } from "@/lib/useSession";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const session = useSession();
  if (session.status === "offline")
    return <p className="p-8 text-sm text-red-600">Can&apos;t reach the server. Check that the API is running, then reload.</p>;
  if (session.status !== "authed") return <p className="p-8 text-sm text-zinc-500">Loading…</p>;
  return (
    <div className="flex h-screen flex-col">
      <Nav session={session} />
      <main className="min-h-0 flex-1">{children}</main>
    </div>
  );
}
