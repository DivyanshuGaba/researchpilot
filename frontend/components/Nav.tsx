"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { logout, type Session } from "@/lib/api";

export default function Nav({ session }: { session: Session }) {
  const path = usePathname();
  const link = (href: string, label: string) => (
    <Link
      href={href}
      className={`rounded px-3 py-1 text-sm ${
        path.startsWith(href)
          ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
          : "hover:bg-zinc-100 dark:hover:bg-zinc-800"
      }`}
    >
      {label}
    </Link>
  );
  return (
    <header className="flex items-center justify-between border-b border-zinc-200 px-4 py-2 dark:border-zinc-800">
      <nav className="flex items-center gap-2">
        <span className="mr-2 font-semibold">ResearchPilot</span>
        {link("/documents", "Documents")}
        {link("/chat", "Chat")}
        {link("/study", "Study")}
      </nav>
      <div className="flex items-center gap-3 text-sm text-zinc-500">
        <span>{session.email}</span>
        <button className="underline" onClick={logout}>
          Sign out
        </button>
      </div>
    </header>
  );
}
