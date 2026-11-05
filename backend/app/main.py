import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_auth, routes_chat, routes_documents, routes_study
from app.config import get_settings


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the embedding model before serving, so the first question is not a cold start."""
    settings = get_settings()
    if settings.warm_embedder:
        from app.services.embeddings import get_embedder

        await run_in_threadpool(get_embedder(settings).embed_query, "warm up")
        from app.services.retrieval.reranker import get_reranker

        reranker = get_reranker(settings)
        if reranker is not None:
            from app.services.vectorstore import Hit

            probe = Hit(uuid.uuid4(), uuid.uuid4(), "", 1, 1, 0.0, "warm up", "warm up")
            await run_in_threadpool(reranker.rerank, "warm up", [probe], 1)
    yield


app = FastAPI(title="ResearchPilot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(routes_auth.router)
app.include_router(routes_documents.router)
app.include_router(routes_chat.router)
app.include_router(routes_study.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
