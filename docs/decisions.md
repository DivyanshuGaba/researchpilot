# Decisions

Ambiguities resolved with the simplest reasonable option (CLAUDE.md section 12).

## Phase 1

- **Dev auth.** `X-User-Id` header plus `POST /auth/dev-user`. Real JWT auth is Phase 5. Tenant isolation does not depend on it: the id is set with `SET LOCAL app.user_id` and enforced by Postgres RLS.
- **Two DB roles.** Migrations run as `postgres` (owner). The app and worker connect as `rp_app`, a non-superuser non-owner, so RLS actually applies (architecture.md 4.3).
- **Token counting is approximate** (about 4 chars per token). Avoids a tokenizer download at runtime and in CI. Revisit if chunk sizes need to be exact.
- **Parent/child chunking from day one**, following architecture.md over CLAUDE.md's flat 800/100 chunker. Children (~250 tokens, 40 overlap) are embedded; parents (~900) are sent to the LLM.
- **Citation markers are `[c_N]`** (architecture.md 5.6), not `[doc_id:page]`. The server maps them to document, page and chunk id and drops unknown markers.
- **Enqueue after commit** instead of the outbox pattern. A crash between commit and enqueue leaves a `queued` document with no job. The outbox dispatcher is a Phase 5 item.
- **No upload dedupe yet.** The file hash is stored on `document_versions` so dedupe can be added without a migration.
- **DOCX and URL documents report page 1.** They have no fixed pagination.
- **Scanned PDFs fail with a clear message.** OCR is an optional later stage.
- **No `temperature` parameter** on the Anthropic call: the installed SDK's `messages.create` does not accept it.
- **Retrieval is vector-only, top 5, no confidence gate** in Phase 1. It refuses only when nothing is retrieved. Hybrid, rerank and gating are Phase 3.

## Phase 2

- **Streaming uses POST + fetch, not EventSource**, because the request needs a body and the `X-User-Id` header. The client parses SSE frames itself (`lib/api.ts`, unit tested).
- **The server's final answer replaces streamed text.** Tokens stream raw, so they can contain citation markers the server later rejects. The `done` event carries the cleaned answer and the UI swaps it in. Unresolved markers show as `[·]` while streaming.
- **Disconnect saves the partial answer** with `messages.incomplete = true` (migration 0002). Pressing Stop in the UI aborts the request and exercises this path.
- **Heartbeat via a producer task and queue**, so idle periods (retrieval, time to first token) still emit `: ping` every 15 s.
- **PDFs are fetched with the auth header and shown from a blob URL**; the file endpoint is tenant-scoped, so another user gets 404.
- **Highlights use the child chunk's PyMuPDF bounding boxes** (points, top-left origin), scaled by rendered width / PDF page width. Rotated pages are not handled yet.
- **Citation page is exact**: children never span pages (found while verifying Phase 1: a child covering pages 1 to 3 cited page 1).
- **Session in `localStorage` via `useSyncExternalStore`**; redirects wait for hydration, otherwise a reload signs the user out.
- **Node 22 for the frontend.** The scaffold's Tailwind v4 pipeline fails on Node 20.14. `@tailwindcss/turbopack` was replaced by the standard `@tailwindcss/postcss`.
- **Upload title stays the user's filename.** Ingestion previously overwrote it with the random stored name; fixed with a regression assertion.
