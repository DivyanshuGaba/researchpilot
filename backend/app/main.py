from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_auth, routes_chat, routes_documents
from app.config import get_settings

app = FastAPI(title="ResearchPilot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(routes_auth.router)
app.include_router(routes_documents.router)
app.include_router(routes_chat.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
