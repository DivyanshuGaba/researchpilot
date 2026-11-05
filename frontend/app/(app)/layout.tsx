"use client";

import Nav from "@/components/Nav";
import { useSession } from "@/lib/useSession";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const session = useSession();
  if (!session) return null;
  return (
    <div className="flex h-screen flex-col">
      <Nav session={session} />
      <main className="min-h-0 flex-1">{children}</main>
    </div>
  );
}
