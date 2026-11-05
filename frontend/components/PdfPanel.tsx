"use client";

import dynamic from "next/dynamic";

// pdf.js touches browser-only globals, so it must never render on the server.
const PdfViewer = dynamic(() => import("./PdfViewer"), {
  ssr: false,
  loading: () => <p className="p-3 text-sm text-zinc-500">Loading viewer…</p>,
});

export default PdfViewer;
