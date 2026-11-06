# Deployment

Status: the Dockerfiles, compose file and CI workflow are written but the images have **not yet been built or run end to end**. Treat the steps below as the intended path and verify on first deploy.

## What runs

| Service | Image | Notes |
|---|---|---|
| postgres | `pgvector/pgvector:pg16` | needs the `vector` extension; migrations create roles `rp_app`, `rp_metrics` |
| redis | `redis:7-alpine` | queue and cache; not needed when `INGEST_MODE=inline` |
| api | `backend/Dockerfile` | FastAPI on 8000, CPU-only torch, non-root |
| worker | `backend/Dockerfile` | Celery ingestion worker; not needed in inline mode |
| frontend | `frontend/Dockerfile` | Next.js on 3000 |
| caddy | `caddy:2-alpine` | `prod` profile only; serves site and `/api/*` on one origin |

## Local full stack

```bash
cp .env.example .env        # set ANTHROPIC_API_KEY and a long random JWT_SECRET
docker compose up --build
```

Production-like, with the proxy: `docker compose --profile prod up --build -d`, with `SITE_ADDRESS=your.domain` set so Caddy gets a TLS certificate.

## Required settings in production

- `ENVIRONMENT=prod`, `COOKIE_SECURE=true`
- `JWT_SECRET`: at least 32 random bytes
- `ANTHROPIC_API_KEY`
- Change the database passwords in the three `*_DATABASE_URL` values
- `ADMIN_EMAILS`: JSON list, e.g. `["you@example.com"]`
- Persistent volume mounted at `UPLOAD_DIR` (`/data/uploads`)

## Render (simplest)

1. Create a Postgres instance with pgvector enabled, and run migrations once (`alembic upgrade head` with `MIGRATION_DATABASE_URL`).
2. Create one Docker web service from `backend/Dockerfile` with `INGEST_MODE=inline` and a persistent disk at `/data`. One service and one disk means no Redis or worker.
3. Create a second web service from `frontend/Dockerfile`.
4. Put both behind one domain (Render path routing or a Caddy service) so the refresh cookie stays same-site.

## Fly.io

- `fly launch` per app (`api`, `frontend`), a Fly volume at `/data` for the API, and Fly Postgres or Neon with pgvector.
- Use `INGEST_MODE=inline` with a single API machine, since a volume attaches to one machine.
- Route `/api/*` to the API through a shared domain or a Caddy app.

## AWS

- ECS Fargate for api, worker and frontend; RDS Postgres with pgvector; ElastiCache Redis; EFS (or S3 with a code change) for uploads shared between api and worker; an ALB with path rules (`/api/*` to api, rest to frontend).
- Use `INGEST_MODE=celery` here, since api and worker are separate tasks.

## Known gaps

- Images unbuilt and the stack untested end to end.
- `.github/workflows` has not run on GitHub.
- The inline mode holds uploads on local disk: one instance only.
