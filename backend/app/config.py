from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineConfig(BaseModel, frozen=True):
    """One object drives production and eval (architecture.md section 1)."""

    chunker: str = "recursive"
    child_tokens: int = 250
    parent_tokens: int = 900
    overlap_tokens: int = 40
    retrieval_mode: str = "vector"
    top_k: int = 5
    prompt_version: str = "v1"


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
    pipeline: PipelineConfig = PipelineConfig()


@lru_cache
def get_settings() -> Settings:
    return Settings()
