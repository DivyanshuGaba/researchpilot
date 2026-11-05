# ResearchPilot

Upload PDFs, DOCX files or URLs and chat with them. Answers are grounded in your documents and carry page-level citations. Design: [docs/architecture.md](docs/architecture.md). Decisions: [docs/decisions.md](docs/decisions.md).

## Status

Phases 1 (foundation and basic RAG) and 2 (frontend, streaming, citations) are implemented. Hybrid retrieval, agents, eval and caching come in later phases.

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

## Frontend

Next.js 16 (App Router), TypeScript, Tailwind v4. Needs **Node 22** (`nvm use`, see `frontend/.nvmrc`): Tailwind v4 requires `require(esm)` support.

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000, API at NEXT_PUBLIC_API_URL (default http://localhost:8000)
npm test && npm run lint && npm run build
```

Pages: `/login` (dev sign-in by email), `/documents` (upload with progress, URL ingestion, live status), `/chat` (streamed answers, clickable `p.N` citation chips that open the PDF at that page with the source passage highlighted). The chat stream is `POST /chat/stream` (SSE events: `status`, `token`, `citation`, `done`, `error`, plus a 15 s heartbeat comment).

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
| sentence-transformers | Local embeddings (`BAAI/bge-small-en-v1.5`), no document text leaves the machine |
| react-pdf (pdf.js) | In-browser PDF viewer; citation highlights are drawn from stored bounding boxes |
| vitest | Frontend unit tests (SSE parser, citation splitter) |
| pymupdf | PDF text with page numbers and bounding boxes |
| python-docx, trafilatura | DOCX and web page extraction |

## Known limitations (Phase 1)

- Dev-only auth; vector-only retrieval with no confidence gate; the production Docker image for the frontend is not built yet (Phase 6); highlights are accurate for unrotated PDF pages only; scanned PDFs are rejected; DOCX and URLs report page 1.

Integration tests use the compose Postgres on host port 5433 (5432 is often taken by a local Postgres): `docker compose up -d postgres`, then `alembic upgrade head` with `MIGRATION_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5433/researchpilot`.
