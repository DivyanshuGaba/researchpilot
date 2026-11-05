from fastapi import FastAPI

from app.api import routes_auth, routes_chat, routes_documents

app = FastAPI(title="ResearchPilot")
app.include_router(routes_auth.router)
app.include_router(routes_documents.router)
app.include_router(routes_chat.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
