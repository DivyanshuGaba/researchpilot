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

## Phase 5

### Auth, limits, reliability
- **Auth:** argon2id passwords; HS256 access tokens that live 15 minutes, with the algorithm pinned on decode and a `typ` claim; refresh tokens are random, stored only as SHA-256, rotated on every use, sent as an HttpOnly `SameSite=Lax` cookie scoped to `/auth`. Presenting an already-used refresh token revokes every session of that user (theft signal). Cookie endpoints also check the `Origin` header. Login says the same thing for a wrong password and an unknown email and does the same hashing work. The access token lives only in browser memory.
- **Accepted trade-offs:** registration reveals whether an email is taken (409) and is open unless `REGISTRATION_ENABLED=false`; no email verification or password reset yet; admins are a list of emails in `ADMIN_EMAILS`.
- **The API refuses to start** outside `ENVIRONMENT=dev` with the default or a short `JWT_SECRET`. The old `X-User-Id` header and `/auth/dev-user` are gone.
- **Limits:** per user 30 model-calling requests a minute (chat and study) and 30 uploads an hour; login 10 attempts per 15 minutes per address and email; 10 sign-ups an hour per address; a daily token budget of 500 000 per user. Counters live in Redis; if Redis is down the limiter falls back to an in-process counter at half the limit and the API stays up. Tokens are counted from every model call in a request (rewrite, summary, answer) through `MeteredLLM`, including study endpoints.
- **Outbox for ingestion:** the upload writes the document and an `outbox` row in one transaction; the API dispatches right after commit and a background loop (5 s) retries failures with `FOR UPDATE SKIP LOCKED`, giving up after 10 attempts (the row then stays pending and shows in the queue-depth metric). Delivery is at least once; ingestion replaces a version's chunks, so a repeat is harmless.
- **Model failures** become `LLMUnavailable` with a safe message and a category (`rate_limited` 429, `timeout` 504, `unavailable` and `misconfigured` and `billing` 503). The SDK already retries 429 and 5xx three times with backoff; calls share one process-wide concurrency cap (8). When generation fails after retrieval succeeded, the response still carries the retrieved passages so the user gets value. A depleted credit balance is a plain 400 from the provider and is now classified as `billing`; that was found when the evaluation runs used up the account.

### Guardrails
- **Layers:** ingestion scanner (normalizes zero-width and full-width characters; 8 pattern families) flags passages; flagged passages are marked `suspicious="true"` with a warning inside the prompt (prompt `v3`, now the default); our own delimiter tags are neutralized inside document text so a document cannot close or forge `<source>`; tools have no write access; the output sanitizer removes markdown and HTML images, drops link destinations to unknown domains, and strips query strings from bare URLs. Flagged passages stay retrievable.
- **The scanner is a heuristic**, tuned to flag 18 attack phrasings and none of 11 ordinary scientific sentences, and it flags nothing in the four evaluation papers. It is a signal for the other layers, not a gate.
- **PII and secrets:** emails, phone numbers, SSNs and Luhn-valid card numbers are masked before query logs and traces; API keys, JWTs and bearer tokens are redacted from every log line by a log record factory. A 16-digit number that fails the card checksum can still be masked as a phone number (over-masking is the safe direction).
- **A model may refuse the whole answer** when a document tells it to fetch the cloud-metadata address (`stop_reason: "refusal"`, seen in a first injection run). That is a denial of answer for that document, not a leak. See the injection results in the Evaluation section.

### Observability
- **`query_logs`** holds one row per answered question: mode, pipeline hash, masked rewritten query, retrieved chunk ids, top score, gate flag, cache status, per-stage timings, tokens, cache-read tokens, cost, degraded stages and any error code. `messages` now carries latency and cost. Logging is best effort and can never fail an answer.
- **Pipeline hash** covers the pipeline config, answer model, confidence threshold, embedding model and reranker; caches and logs share it.
- **`/admin/metrics`** reads aggregates through a separate read-only Postgres role (`rp_metrics`, `BYPASSRLS`, `SELECT` only) so the application role stays bound by row level security. `/usage` shows a user their own cost.
- **Langfuse export** goes through the public ingestion API with masked inputs and fails soft. It has not been run against a real Langfuse project.
- **`prompt_cached` is recorded but unproven:** the system prompt is marked cacheable, but it is likely shorter than the model's minimum cacheable prefix, so cache reads may never occur. The `cache_read_tokens` column will show it.
- **Not built from the architecture's alert list:** alerting on p95 latency, refusal spikes and daily cost (the metrics exist, the alerts do not).

### Caching
- **Layers:** embedding cache (Redis, keyed by model and text), exact response cache, semantic cache, prompt cache. Keys carry the user, a hash of the document set's latest versions, the pipeline hash, the mode and the normalized standalone query. Adding, re-uploading or removing a document changes the key, so a stale answer cannot be served. Agent answers, degraded runs, empty answers and model-failure messages are never stored. Redis down means the cache is skipped for 30 s at a time; requests continue.
- **The semantic threshold is 0.97, not the architecture's 0.95.** The evaluation found that at 0.95 one of 12 near-duplicate questions with different answers (replace with `[MASK]` versus replace with a random token, similarity 0.956) would be answered with the wrong cached answer, while 0.97 had none and still reused half of the true paraphrases. The sample is small (12 and 8 pairs); the threshold is configurable.

### Evaluation
- **Corpus and questions:** four arXiv papers pinned by hash, 49 questions (41 answerable: 30 factual, 4 multi-hop, 4 table, 3 cross-document; 8 unanswerable). Each answer has a verbatim evidence phrase verified on its page; unanswerable terms are verified absent.
- **Judge:** `claude-opus-5-5` judging `claude-sonnet-5-5` answers against the sources the answer was given. The installed API rejects sampling parameters, so the judge cannot be pinned to temperature 0. Spot check: 7 of 49 verdicts (14%) read by hand, including every non-perfect one, 7 of 7 agreed. Small sample, one reader.
- **Reading the numbers** (all five generation configurations ran): with 41 answerable questions one question is 0.024 and differences under about 0.05 to 0.1 are noise. **Reranking is the consistent winner**: hybrid + rerank scored correctness 0.95 and false refusals 0.024 at 800 tokens, against 0.90 / 0.073 for vector search and 0.87 / 0.073 for hybrid without reranking, and it raised MRR in every chunking setting. Hybrid search alone was not reliably better than vector search. 800 and 1200-token sections tied on answer correctness (0.95 each) even though 1200 retrieved slightly better (hit@5 0.93 vs 0.88, within noise). Semantic chunking did not beat recursive chunking at the same size (0.93 vs 0.95). Faithfulness and citation precision were at or near 1.0 everywhere, refusal accuracy was 1.0 everywhere, and an answer costs about $0.012 (the judge adds about $0.024). **Defaults were not changed**: the shipped pipeline (hybrid + rerank, recursive chunks of 250/900 tokens) is the configuration these results favour, and the remaining differences are within noise.
- **Prompt injection (real model, 3 trials each):** 5 of 6 attacks were answered correctly with nothing obeyed and nothing leaked (override, obfuscated, fake system message, delimiter escape, exfiltration link). The attack that tells the model to fetch the cloud-metadata address made the model refuse to answer in 3 of 3 trials: a denial of answer for that document, not a leak. The first run exposed that the scanner missed the "include this link" attack; a `link_injection` pattern was added and the rerun flags it. One model, one question, small trials: this shows the layers work together on these attacks, not that the model is immune.
- **CI smoke check:** `python -m eval.run_eval --smoke` replays recorded model replies (no API calls) for 10 questions and fails if hit@5, recall@5, MRR, faithfulness, correctness or refusal accuracy fall below `eval/baseline.json`. A changed prompt, chunking or retrieval makes a recording miss, which also fails with an instruction to re-record. A CI workflow that runs it is Phase 6.
- **Generation latency** of the two earliest runs was discarded because the timer included queueing; the other three use the fixed timer (4 questions in flight).

### Bugs found by running real data and real browsers this phase
- PDF text from equations contains NUL bytes; Postgres rejected them and ingestion of the RAG paper failed. Control characters are now stripped in every parser.
- Credentialed CORS was missing, which would have broken the refresh cookie in any browser. Fixed and tested.
- A fresh `docker compose up` would have failed at migration 0002 (found last phase); verified again with migrations 0005 and 0006.
- Evaluation measurement bugs of my own: a canary list that counted *describing* an attack as obeying it, and a generation timer that included queueing. Both fixed; the affected numbers were removed from the report.

### Not done
Dedupe by file hash, intent routing, MMR, embedding-model migration tooling, query-log retention, JWT key rotation, email verification and password reset, a per-user usage page in the UI (the endpoint exists), and the architecture's alerts.
