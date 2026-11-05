import logging

from app.config import Settings
from app.services.generation.llm_client import LLMClient, LLMRequest
from app.services.generation.prompts import (
    COMPARE_REWRITE_PROMPT_V1,
    REWRITE_PROMPT_V1,
    render_history,
)
from app.services.memory import HistoryState

log = logging.getLogger(__name__)
MAX_QUERY_CHARS = 500


def fallback_query(question: str, state: HistoryState) -> str:
    """If the rewrite call fails: the raw question plus the last user turn (architecture 13)."""
    for role, text in reversed(state.recent):
        if role == "user":
            return f"{text} {question}"[:MAX_QUERY_CHARS]
    return question


async def rewrite_query(
    question: str,
    state: HistoryState,
    summary: str,
    llm: LLMClient,
    settings: Settings,
    mode: str = "qa",
) -> tuple[str, bool]:
    """Turn a follow-up into a standalone query. Returns (query, used_llm).

    In QA mode the first turn has nothing to resolve, so the LLM call is skipped. Compare mode
    always rewrites: comparison wording ("how do these differ on X") makes a cross-encoder score
    every document's passage about X as irrelevant, so each document is searched for the topic.
    """
    has_history = bool(state.recent or summary)
    if not has_history and mode != "compare":
        return question, False
    history = render_history(summary, state.recent)
    user = f"{history}\n\nLatest message: {question}" if history else f"Request: {question}"
    try:
        resp = await llm.complete(
            LLMRequest(
                system=COMPARE_REWRITE_PROMPT_V1 if mode == "compare" else REWRITE_PROMPT_V1,
                user=user,
                model=settings.pipeline.rewrite_model,
                max_tokens=150,
            )
        )
        text = resp.text.strip().strip("\"'")
    except Exception:
        log.exception("query rewrite failed")
        return fallback_query(question, state), False
    if not text or len(text) > MAX_QUERY_CHARS:
        return fallback_query(question, state), True
    return text, True
