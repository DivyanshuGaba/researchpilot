from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineConfig(BaseModel, frozen=True):
    """One object drives production and eval (architecture.md section 1)."""

    chunker: str = "recursive"  # recursive | semantic
    child_tokens: int = 250
    parent_tokens: int = 900
    overlap_tokens: int = 40
    semantic_percentile: int = 25  # split where adjacent-sentence similarity is in the lowest N%
    retrieval_mode: str = "hybrid_rerank"  # vector | hybrid | hybrid_rerank
    dense_k: int = 30
    sparse_k: int = 30
    rrf_k: int = 60
    rerank_candidates: int = 30
    top_k: int = 6  # final number of child hits kept
    max_per_parent: int = 3
    context_token_budget: int = 6000
    compare_per_doc: int = 3
    history_messages: int = 6
    summarize_batch: int = 4
    rewrite_model: str = "claude-haiku-5-5"
    prompt_version: str = "v2"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5-5"
    database_url: str = "postgresql+psycopg://rp_app:rp_app@localhost:5432/researchpilot"
    migration_database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/researchpilot"
    redis_url: str = "redis://localhost:6379/0"
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    reranker: str = "cross_encoder"
    jwt_secret: str = "change_me"
    retrieval_min_score: float = 0.35
    upload_dir: str = "./storage/uploads"
    max_upload_mb: int = 50
    warm_embedder: bool = True
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    pipeline: PipelineConfig = PipelineConfig()


@lru_cache
def get_settings() -> Settings:
    return Settings()
