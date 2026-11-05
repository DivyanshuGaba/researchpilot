import logging

from app.config import Settings
from app.services.generation.llm_client import LLMClient, LLMRequest
from app.services.generation.prompts import REWRITE_PROMPT_V1, render_history
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
    question: str, state: HistoryState, summary: str, llm: LLMClient, settings: Settings
) -> tuple[str, bool]:
    """Turn a follow-up into a standalone query. Returns (query, used_llm).

    The first turn of a conversation has nothing to resolve, so the LLM call is skipped.
    """
    if not state.recent and not summary:
        return question, False
    user = f"{render_history(summary, state.recent)}\n\nLatest message: {question}"
    try:
        resp = await llm.complete(
            LLMRequest(
                system=REWRITE_PROMPT_V1,
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
