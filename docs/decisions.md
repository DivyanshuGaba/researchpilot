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
