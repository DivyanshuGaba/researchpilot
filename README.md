# ResearchPilot

Upload PDFs, DOCX files or URLs and chat with them. Answers are grounded in your documents and carry page-level citations. Design: [docs/architecture.md](docs/architecture.md). Decisions: [docs/decisions.md](docs/decisions.md).

## Status

Phases 1 to 5 are implemented: ingestion, streaming chat with page-level citations and PDF highlights, hybrid retrieval with reranking and a confidence gate, query rewriting with conversation memory, multi-document comparison, a semantic chunker, a bounded tool-using agent, and quiz, flashcard and summary generation. Phase 5 added real auth, rate limits and quotas, guardrails, observability, caching and the evaluation harness; CI, production images and deployment notes are Phase 6.

## Quick start

```bash
cp .env.example .env        # add ANTHROPIC_API_KEY
docker compose build
docker compose up -d
```

Open the frontend (below), create an account and upload a PDF. Or from the command line:

```bash
curl -s -X POST localhost:8000/auth/register -H 'content-type: application/json' \
  -d '{"email":"me@example.com","password":"a-long-passphrase"}' -c cookies.txt   # returns access_token
TOKEN=...   # paste the access_token
curl -X POST localhost:8000/documents -H "Authorization: Bearer $TOKEN" -F file=@paper.pdf
curl localhost:8000/documents -H "Authorization: Bearer $TOKEN"        # wait for status "ready"
curl -X POST localhost:8000/chat -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"question":"What method did the authors use?"}'
```

Set a real `JWT_SECRET` before anything but local use (the API refuses to start otherwise when `ENVIRONMENT` is not `dev`), and list admin emails in `ADMIN_EMAILS` to see `/metrics`.

## How a question is answered

1. **Memory and rewrite:** the last 6 messages (older ones summarized) turn a follow-up like "where does it happen?" into a standalone query. Skipped on the first turn.
2. **Retrieve:** dense vector search and Postgres full-text search, merged with Reciprocal Rank Fusion.
3. **Rerank:** a cross-encoder rescores the top 30; at most 3 hits per section are kept.
4. **Confidence gate:** if the best rerank score is below `RETRIEVAL_MIN_SCORE`, the answer is "your documents don't cover this", with no model call and no citations, plus the closest passages to inspect.
5. **Generate:** sources go to Claude wrapped as untrusted data; `[c_N]` citations are validated server-side against what was retrieved.

Compare mode (select two or more documents and tick Compare) retrieves per document and answers with a section per document. Settings: `PIPELINE__RETRIEVAL_MODE` (`vector`, `hybrid`, `hybrid_rerank`), `PIPELINE__CHUNKER` (`recursive`, `semantic`), `RETRIEVAL_MIN_SCORE`. Rationale and measurements: [docs/decisions.md](docs/decisions.md).

## Security, caching and operations

- **Auth and limits:** argon2id passwords, 15 minute access tokens held in memory, rotating HttpOnly refresh cookie with reuse detection, per-user rate limits and a daily token budget in Redis (in-process fallback if Redis is down), tenant isolation by Postgres row level security. Ingestion jobs go through an outbox so a crash cannot lose them.
- **Guardrails:** injection scanner, delimiter neutralizing and suspicious-passage marking in the prompt, an output sanitizer against exfiltration through images and links, PII masking in logs and traces, secret redaction on every log line.
- **Caching:** embedding, exact and semantic response caches (Redis), scoped by user, document-set version and pipeline hash; agent and degraded answers are never cached. The semantic threshold is 0.97 because the evaluation found wrong-answer hits at 0.95.
- **Observability:** a `query_logs` row per question (stage timings, tokens, cost, cache status, errors), cost on every message, `/usage` for the signed-in user and `/admin/metrics` plus the `/metrics` page for admins. Optional Langfuse export.

## Evaluation

`backend/eval/` holds a 49 question set over four open-access papers, the metrics, an LLM judge and sweep tooling. Full results and caveats: [docs/eval_results.md](docs/eval_results.md).

```bash
cd backend
python -m eval.fetch_docs && python -m eval.verify_dataset   # download pinned PDFs, check every answer's evidence
python -m eval.sweep                                          # retrieval grid, no model calls
python -m eval.sweep --skip-retrieval --generate recursive-800-hybrid_rerank,recursive-800-vector
python -m eval.report                                         # rebuild docs/eval_results.md
python -m eval.adversarial cache                              # semantic cache false-hit study
python -m eval.adversarial injection                          # prompt injection against the real model (cents)
python -m eval.run_eval --smoke                               # 10 questions, recorded replies, free; fails below eval/baseline.json
```

Headline results (41 answerable questions, so one question is 0.024 and small gaps are noise): **reranking is the consistent winner**. Hybrid + rerank reached correctness 0.95 and 2% false refusals, against 0.90 / 7% for vector search and 0.87 / 7% for hybrid without reranking; retrieval hit@5 was 0.83 (vector) vs 0.88 (hybrid + rerank, 800-token sections) and 0.93 (1200). Faithfulness was about 1.0 and refusal accuracy 1.0 throughout. An answer costs about $0.012. 5 of 6 prompt-injection attacks were fully resisted by a real model, and the sixth made it refuse rather than leak. The semantic cache threshold is 0.97 because 0.95 gave a wrong-answer hit.

## Agent and study tools

- **Agent mode** (`mode: "agent"`, the Agent toggle in chat): the model can call `search_documents`, `calculator` and `fetch_url`, at most 5 rounds and 8 calls, within a token and time budget, then answers. Tool calls are shown live. `fetch_url` only reaches public hosts (every redirect is re-checked), and everything a tool returns is treated as untrusted data. Details and the one known gap: [docs/decisions.md](docs/decisions.md).
- **Study endpoints** (`/study`): `POST /study/quizzes`, `POST /study/flashcards`, `POST /study/summaries`, plus `GET` for saved quizzes and flashcards and for cached summaries. Quiz and flashcard sources point to the exact page and highlight the passage; summaries are map-reduce and cached per document version.

## Frontend

Next.js 16 (App Router), TypeScript, Tailwind v4. Needs **Node 22** (`nvm use`, see `frontend/.nvmrc`): Tailwind v4 requires `require(esm)` support.

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000, API at NEXT_PUBLIC_API_URL (default http://localhost:8000)
npm test && npm run lint && npm run build
```

Pages: `/login` (dev sign-in by email), `/documents` (upload with progress, URL ingestion, live status), `/chat` (streamed answers, clickable `p.N` citation chips that open the PDF at that page with the source passage highlighted; Compare and Agent toggles), `/study` (interactive quiz, flip-card flashcards, summary with clickable page references). The chat stream is `POST /chat/stream` (SSE events: `status`, `token`, `citation`, `done`, `error`, plus a 15 s heartbeat comment).

## Development

```bash
cd backend
pip install -e .[dev]
pytest && ruff check . && mypy app
```

## Dependencies and why

| Package | Why |
|---|---|
| fastapi, uvicorn, pydantic | API and validation |
| sqlalchemy, alembic, psycopg, pgvector | Postgres access, migrations, vector column |
| celery, redis | Background ingestion |
| anthropic | Answer generation |
| sentence-transformers | Local embeddings (`BAAI/bge-small-en-v1.5`) and the cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`); no document text leaves the machine |
| react-pdf (pdf.js) | In-browser PDF viewer; citation highlights are drawn from stored bounding boxes |
| vitest | Frontend unit tests (SSE parser, citation splitter) |
| argon2-cffi, PyJWT, email-validator | Password hashing, access tokens, address validation |
| httpx | `fetch_url` tool (manual redirect handling so each hop can be checked) |
| pymupdf | PDF text with page numbers and bounding boxes |
| python-docx, trafilatura | DOCX and web page extraction |

## Known limitations (Phase 1)

- The confidence threshold and semantic-cache threshold come from small samples; no email verification or password reset; CI workflow not written yet (Phase 6); no intent routing (compare and agent are manual toggles); no rate limits or quotas, so the agent and study endpoints call the model on every request; `fetch_url` does not pin the connection to the validated IP (DNS rebinding gap); answers render as plain text (markdown headings show as `##`); the production Docker image for the frontend is not built yet (Phase 6); highlights are accurate for unrotated PDF pages only; scanned PDFs are rejected; DOCX and URLs report page 1.

Integration tests use the compose Postgres on host port 5433 (5432 is often taken by a local Postgres): `docker compose up -d postgres`, then `alembic upgrade head` with `MIGRATION_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5433/researchpilot`.
