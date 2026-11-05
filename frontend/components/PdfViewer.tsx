"use client";

import { useEffect, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";
import { fetchPdf, type BBox } from "@/lib/api";

// Must be set in the module that renders <Document>/<Page> (react-pdf README).
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

const WIDTH = 520;

type Props = { documentId: string; initialPage: number; highlights: BBox[]; title: string };

export default function PdfViewer({ documentId, initialPage, highlights, title }: Props) {
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [numPages, setNumPages] = useState(0);
  const [pageNo, setPageNo] = useState(initialPage);
  const [size, setSize] = useState<{ w: number } | null>(null);

  useEffect(() => {
    let objectUrl: string | null = null;
    let cancelled = false;
    fetchPdf(documentId)
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      })
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [documentId]);

  const scale = size ? WIDTH / size.w : 1;
  const onPage = highlights.filter((h) => h.page === pageNo);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-2 border-b border-zinc-200 px-3 py-2 text-sm dark:border-zinc-800">
        <span className="truncate font-medium" title={title}>
          {title}
        </span>
        <span className="flex items-center gap-1 whitespace-nowrap">
          <button
            className="rounded px-2 py-0.5 hover:bg-zinc-100 disabled:opacity-40 dark:hover:bg-zinc-800"
            disabled={pageNo <= 1}
            onClick={() => setPageNo((p) => p - 1)}
            aria-label="Previous page"
          >
            ←
          </button>
          <span>
            {pageNo} / {numPages || "…"}
          </span>
          <button
            className="rounded px-2 py-0.5 hover:bg-zinc-100 disabled:opacity-40 dark:hover:bg-zinc-800"
            disabled={pageNo >= numPages}
            onClick={() => setPageNo((p) => p + 1)}
            aria-label="Next page"
          >
            →
          </button>
        </span>
      </div>
      <div className="flex-1 overflow-auto bg-zinc-100 p-3 dark:bg-zinc-900">
        {error && <p className="text-sm text-red-600">{error}</p>}
        {!error && !url && <p className="text-sm text-zinc-500">Loading PDF…</p>}
        {url && (
          <Document
            file={url}
            onLoadSuccess={(d) => setNumPages(d.numPages)}
            onLoadError={(e) => setError(e.message)}
            loading={<p className="text-sm text-zinc-500">Rendering…</p>}
          >
            <div className="relative mx-auto w-fit shadow">
              <Page
                pageNumber={pageNo}
                width={WIDTH}
                onLoadSuccess={(p) => setSize({ w: p.originalWidth })}
              />
              {size &&
                onPage.map((h, i) => (
                  <div
                    key={i}
                    data-testid="highlight"
                    className="pointer-events-none absolute rounded-sm bg-yellow-300/50 mix-blend-multiply ring-1 ring-yellow-500/60"
                    style={{
                      left: h.bbox[0] * scale,
                      top: h.bbox[1] * scale,
                      width: (h.bbox[2] - h.bbox[0]) * scale,
                      height: (h.bbox[3] - h.bbox[1]) * scale,
                    }}
                  />
                ))}
            </div>
          </Document>
        )}
      </div>
    </div>
  );
}
