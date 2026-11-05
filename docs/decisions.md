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

## Real-stack verification (end of Phase 2)

Ran the real Celery worker (`--pool=solo`; fork plus torch is risky on macOS), real `bge-small` embedder, real Postgres and real Claude (`claude-sonnet-5-5`):

- Upload to `ready` in about 7.5 s for a 3 page PDF; correct page counts and title.
- Streaming answers were grounded, cited the right page with bounding boxes, and declined to invent an unsupported "why".
- Client disconnect mid-stream saved the partial answer with `incomplete = true`.
- No API key or traceback in the API or worker logs.
- **Embedder warm-up at API start** (`WARM_EMBEDDER`, default on). The first question previously paid about 5 s for model loading.
- **Known weakness, for Phase 3:** an unanswerable question ("capital of Mongolia") still retrieves the nearest chunk, and the model's refusal carries a `[c_1]` citation. The confidence gate (rerank score vs `RETRIEVAL_MIN_SCORE`) is what should short-circuit this before the LLM and return no citation.

## Phase 3

- **Retrieval pipeline** (`retrieval/service.py`): dense top 30 + Postgres full-text top 30 -> RRF (k=60) -> cross-encoder rerank of the fused top 30 -> cap 3 hits per parent -> keep top 6 -> confidence gate. Modes `vector`, `hybrid`, `hybrid_rerank` (default) so the Phase 5 eval can sweep them. Dense and sparse run sequentially, not in parallel (about 10 ms total on this corpus).
- **Keyword query is an OR query.** `websearch_to_tsquery` ANDs every word, so a full question matches nothing. Words are sanitized to `[a-z0-9]+` and OR-ed; the `english` config drops stopwords.
- **Reranker scores get an explicit sigmoid.** Measured with the installed sentence-transformers: `predict()` returns raw logits (relevant about -1 to 8, irrelevant about -11), not 0..1. Without the sigmoid, the original 0.35 threshold falsely refused 2 of 8 answerable questions.
- **`RETRIEVAL_MIN_SCORE` is now 0.05** (provisional). After the sigmoid, 8 answerable questions scored 0.26 to 1.0 and 6 unanswerable ones scored about 0.0, all refused, in about 0.1 s with no LLM call. This is a 14 question, 2 document sample; recalibrate on the Phase 5 eval set. The gate only applies to reranker scores; if the reranker is unavailable the request is marked `degraded` and answers without a gate.
- **A gated answer returns no citations**, but includes the 3 closest passages so the user can inspect them (architecture 5.4). The UI shows them as clickable items that open the PDF.
- **Compare mode is explicit** (`mode: "compare"` plus at least two `document_ids`), not intent-routed; the router from architecture 5.2 is not built yet. Retrieval runs per document (top 3 each), sources are grouped in `<document>` blocks, and a document with no relevant passage is stated as such.
- **Compare mode always rewrites the query into a neutral topic.** Found in real use: "how do the two documents differ on ATP yield" scored 0.027 against a passage stating the ATP yield (gated), while "ATP yield per glucose molecule" scored 0.995. Comparison wording makes a cross-encoder rate every passage as irrelevant.
- **Query rewrite uses Haiku** (`PIPELINE__REWRITE_MODEL`) and is skipped on the first QA turn. If it fails, the query falls back to the last user turn plus the question.
- **Memory:** last 6 messages verbatim; once more than 10 are unsummarized, the older ones are folded into `conversations.summary` (migration 0003) using Haiku. If summarization fails the older messages are dropped for that turn (marked `degraded`). History is stripped of citation markers and the system prompt says it is not evidence.
- **Message ordering** uses `(created_at, role DESC)` because a user/assistant pair is inserted in one transaction and shares a timestamp.
- **Semantic chunker** (`PIPELINE__CHUNKER=semantic`): breaks children where adjacent-sentence similarity is in the lowest 25%, never mid-page, with a minimum child size and no overlap. On the 3 page test PDF it produced the same chunks as the recursive chunker, so it is verified to run but not shown to be better; that is for the Phase 5 eval.
- **Prompts:** `v2` (default) adds the rule that conversation history is not evidence and that a source must not be cited for something it did not answer. `v1` is kept. Compare uses its own prompt.
- **Pipeline settings from the environment** via `PIPELINE__<FIELD>` (e.g. `PIPELINE__RETRIEVAL_MODE=vector`).
- **/chat and /chat/stream share one code path** (`chat_service.prepare_turn`). The non-streaming route previously duplicated retrieval and persistence.
- **Not done from the architecture doc:** intent routing, MMR (only the per-parent cap), embedding cache, and parallel dense/sparse.

## Phase 4

- **Agent mode is explicit** (`mode: "agent"`, a toggle in the UI), not router-selected; plain Q&A never enters the loop (architecture section 6). It is mutually exclusive with compare mode.
- **Limits** (`AgentLimits`): 5 tool rounds, 8 tool calls in total (so a model cannot fan out with parallel calls), 60 000 tokens, 60 s wall clock, 30 s per tool. When any limit is hit the model gets one last call with `tool_choice: none` and the answer is suffixed "(Stopped early: ...)". The API requires `tools` to be present whenever the history contains tool blocks, which is why the wrap-up passes them with `tool_choice: none` instead of omitting them.
- **Every tool call is logged and streamed** as a `tool` SSE event (started/finished, duration, output size, error flag), returned in the response meta, and stored on the message (`messages.tool_calls`). Inputs are truncated to 200 characters in logs and events; document text is never logged.
- **The agent's answer is not token-streamed.** Tools run first, then the whole answer arrives as one `token` event followed by citation events. Streaming the final turn is a possible later polish.
- **`search_documents` keeps the rerank and confidence gate** (a gated search returns "no relevant passages") and shares one `c_N` numbering across calls; the model may narrow the user's document scope but never widen it.
- **Calculator is an AST whitelist** (numbers, `+ - * / // % **`, a fixed function list, `pi`/`e`): no names, attributes, calls to anything else, booleans or strings. Exponent, bit-length, expression length and nesting depth are capped; 18 hostile inputs are tested.
- **`fetch_url`**: public hosts only, http(s) only, redirects followed by hand and re-checked at every hop (max 3), 1 MB cap, 10 s timeout, text/html or text/plain only, main text extracted with trafilatura (scripts and navigation dropped), output clipped to 6 000 characters and wrapped in `<untrusted_web_content>`. **Known gap:** the filter resolves DNS and the HTTP client resolves it again, so a hostile DNS server could rebind between the two. Pinning the connection to the validated IP is the fix and is not done.
- **A model refusal can come back as empty content** (`stop_reason: "refusal"`; seen for a request to fetch the cloud metadata address). Empty replies now become an explicit message in all three answer paths instead of a blank bubble.
- **Caching rule for Phase 5:** agent answers that used `fetch_url` must not be cached (architecture section 7). There is no cache yet.
- **Quiz and flashcards sample from child chunks**, one passage per requested question (up to 12), evenly spaced across the document (not random). Child chunks never span pages, so the cited page is exact, and their stored bounding boxes let the UI highlight the source. Sampling parents first gave "p.1" for a question that was on page 3 in a short document.
- **Structured output = prompt + Pydantic**, not provider schema enforcement: 4 distinct options, `answer_index` 0..3, a source id that must be one of the passages shown. Invalid output is retried once with the validation error appended; a second failure is a safe 502 with no model text in it. Fewer questions than requested are accepted and the UI says so.
- **Summaries are map-reduce**: one Haiku call per section (parent chunk), 5 in parallel, then a reduce with the main model; if the section summaries exceed 6 000 tokens they are reduced in groups first. Beyond 150 sections the document is sampled evenly and flagged `truncated`. Section and document summaries are cached per document version in `summaries` (partial unique indexes plus `ON CONFLICT DO NOTHING` for concurrent requests); a new version never reuses them.
- **Migrations fixed for fresh installs.** 0001 used the live models for `create_all`, so a fresh database would have failed at 0002 with "column already exists". 0001 now creates only the original tables and 0002 to 0004 use `IF NOT EXISTS`. Verified from an empty database and incrementally.
- **No per-user quotas or rate limits yet** on the study endpoints or the agent; they call the model on every request. That is Phase 5 (auth and rate limiting).
