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
    rewrite_model: str = "claude-haiku-5-5"  # also the cheap model for summary map steps
    quiz_max_passages: int = 12
    summary_max_sections: int = 150
    summary_concurrency: int = 5
    summary_reduce_tokens: int = 6000
    prompt_version: str = "v3"


# USD per million tokens (cached: 2026-10-06, Anthropic first-party API). Cache reads cost 0.1x
# and cache writes 1.25x the input price. Update when prices change; unknown models cost 0.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-1": (10.0, 50.0),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_nested_delimiter="__")

    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5-5"
    database_url: str = "postgresql+psycopg://rp_app:rp_app@localhost:5432/researchpilot"
    migration_database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/researchpilot"
    redis_url: str = "redis://localhost:6379/0"
    # Read-only role that may see all tenants, used only by the admin metrics endpoint.
    metrics_database_url: str = "postgresql+psycopg://rp_metrics:rp_metrics@localhost:5432/researchpilot"
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    reranker: str = "cross_encoder"
    jwt_secret: str = "change_me"
    # Provisional: measured on a tiny corpus (relevant >= 0.26, irrelevant <= 0.0001 after
    # sigmoid). Recalibrate on the eval set in Phase 5.
    retrieval_min_score: float = 0.05
    upload_dir: str = "./storage/uploads"
    max_upload_mb: int = 50
    warm_embedder: bool = True
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    pipeline: PipelineConfig = PipelineConfig()

    # --- auth, limits, ops (Phase 5) ---
    environment: str = "dev"  # anything else refuses to start with the default JWT secret
    access_token_minutes: int = 15
    refresh_token_days: int = 14
    cookie_secure: bool = False  # set true behind HTTPS
    registration_enabled: bool = True
    admin_emails: list[str] = []
    rate_limit_per_minute: int = 30  # model-calling endpoints, per user
    login_attempts_per_15min: int = 10  # per client IP and email
    daily_token_budget: int = 500_000  # per user; 0 disables
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    # 0.95 (the architecture's starting point) served a wrong cached answer for 1 of 12
    # near-duplicate questions with different answers (BERT-BASE vs BERT-LARGE style pairs);
    # 0.97 had none while still catching half of the true paraphrases. See docs/eval_results.md.
    semantic_cache_threshold: float = 0.97
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""
    mask_pii_in_logs: bool = True
    llm_max_concurrency: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
