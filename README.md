# ResearchPilot

Upload PDFs, DOCX files or URLs and chat with them. Answers are grounded in your documents and carry page-level citations. Design: [docs/architecture.md](docs/architecture.md). Decisions: [docs/decisions.md](docs/decisions.md).

## Status

Phases 1 to 4 are implemented: ingestion, streaming chat with page-level citations and PDF highlights, hybrid retrieval with reranking and a confidence gate, query rewriting with conversation memory, multi-document comparison, a semantic chunker, a bounded tool-using agent, and quiz, flashcard and summary generation. The evaluation harness, caching, real auth and observability come in Phase 5.

## Quick start

```bash
cp .env.example .env        # add ANTHROPIC_API_KEY
docker compose build
docker compose up -d
```

Create a dev user, upload a PDF, ask a question:

```bash
curl -X POST localhost:8000/auth/dev-user -H 'content-type: application/json' -d '{"email":"me@example.com"}'
# use the returned id below
curl -X POST localhost:8000/documents -H "X-User-Id: $UID" -F file=@paper.pdf
curl localhost:8000/documents -H "X-User-Id: $UID"          # wait for status "ready"
curl -X POST localhost:8000/chat -H "X-User-Id: $UID" -H 'content-type: application/json' \
  -d '{"question":"What method did the authors use?"}'
```

## How a question is answered

1. **Memory and rewrite:** the last 6 messages (older ones summarized) turn a follow-up like "where does it happen?" into a standalone query. Skipped on the first turn.
2. **Retrieve:** dense vector search and Postgres full-text search, merged with Reciprocal Rank Fusion.
3. **Rerank:** a cross-encoder rescores the top 30; at most 3 hits per section are kept.
4. **Confidence gate:** if the best rerank score is below `RETRIEVAL_MIN_SCORE`, the answer is "your documents don't cover this", with no model call and no citations, plus the closest passages to inspect.
5. **Generate:** sources go to Claude wrapped as untrusted data; `[c_N]` citations are validated server-side against what was retrieved.

Compare mode (select two or more documents and tick Compare) retrieves per document and answers with a section per document. Settings: `PIPELINE__RETRIEVAL_MODE` (`vector`, `hybrid`, `hybrid_rerank`), `PIPELINE__CHUNKER` (`recursive`, `semantic`), `RETRIEVAL_MIN_SCORE`. Rationale and measurements: [docs/decisions.md](docs/decisions.md).

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
| httpx | `fetch_url` tool (manual redirect handling so each hop can be checked) |
| pymupdf | PDF text with page numbers and bounding boxes |
| python-docx, trafilatura | DOCX and web page extraction |

## Known limitations (Phase 1)

- Dev-only auth; the confidence threshold is calibrated on a tiny sample; no intent routing (compare and agent are manual toggles); no rate limits or quotas, so the agent and study endpoints call the model on every request; `fetch_url` does not pin the connection to the validated IP (DNS rebinding gap); answers render as plain text (markdown headings show as `##`); the production Docker image for the frontend is not built yet (Phase 6); highlights are accurate for unrotated PDF pages only; scanned PDFs are rejected; DOCX and URLs report page 1.

Integration tests use the compose Postgres on host port 5433 (5432 is often taken by a local Postgres): `docker compose up -d postgres`, then `alembic upgrade head` with `MIGRATION_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5433/researchpilot`.
