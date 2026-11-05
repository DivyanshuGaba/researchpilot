# ResearchPilot

Upload PDFs, DOCX files or URLs and chat with them. Answers are grounded in your documents and carry page-level citations. Design: [docs/architecture.md](docs/architecture.md). Decisions: [docs/decisions.md](docs/decisions.md).

## Status

Phase 1 (foundation and basic RAG) is implemented. Frontend, streaming, hybrid retrieval, agents, eval and caching come in later phases.

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
| pymupdf | PDF text with page numbers and bounding boxes |
| python-docx, trafilatura | DOCX and web page extraction |

## Known limitations (Phase 1)

- Dev-only auth; vector-only retrieval with no confidence gate; no streaming; scanned PDFs are rejected; DOCX and URLs report page 1.

Integration tests use the compose Postgres on host port 5433 (5432 is often taken by a local Postgres): `docker compose up -d postgres`, then `alembic upgrade head` with `MIGRATION_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5433/researchpilot`.
