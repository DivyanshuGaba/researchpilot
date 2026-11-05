"""Multi-turn memory: the last N messages verbatim, older ones folded into a rolling summary."""

import logging
import re
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import PipelineConfig, Settings
from app.models import Conversation, Message
from app.services.generation.llm_client import LLMClient, LLMRequest
from app.services.generation.prompts import SUMMARY_PROMPT_V1, render_history

log = logging.getLogger(__name__)
_MARKER = re.compile(r"\s*\[c_\d+\]")
MAX_ASSISTANT_CHARS = 600


@dataclass
class HistoryState:
    conversation_id: uuid.UUID | None = None
    summary: str = ""
    summarized_count: int = 0
    recent: list[tuple[str, str]] = field(default_factory=list)  # (role, text)
    to_fold: list[tuple[str, str]] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.summary and not self.recent and not self.to_fold


def clean(role: str, text: str) -> str:
    """Strip citation markers and trim long answers; history is for reference, not evidence."""
    if role == "assistant":
        text = _MARKER.sub("", text)
        if len(text) > MAX_ASSISTANT_CHARS:
            text = text[:MAX_ASSISTANT_CHARS].rstrip() + "…"
    return text.strip()


def load_history(
    session: Session, conversation_id: uuid.UUID | None, cfg: PipelineConfig
) -> HistoryState:
    if conversation_id is None:
        return HistoryState()
    conv = session.get(Conversation, conversation_id)
    if conv is None:  # missing, or another tenant's (RLS hides it)
        return HistoryState()
    # created_at ties inside one transaction (a user/assistant pair): 'user' sorts after
    # 'assistant', so descending role puts the question first.
    rows = session.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at, Message.role.desc())
    ).all()
    pending = [(m.role, clean(m.role, m.content)) for m in rows][conv.summarized_count :]
    n = cfg.history_messages
    if len(pending) > n + cfg.summarize_batch:
        return HistoryState(
            conversation_id, conv.summary, conv.summarized_count, pending[-n:], pending[:-n]
        )
    return HistoryState(conversation_id, conv.summary, conv.summarized_count, pending, [])


def save_summary(
    session: Session, conversation_id: uuid.UUID, summary: str, summarized_count: int
) -> None:
    conv = session.get(Conversation, conversation_id)
    if conv is not None:
        conv.summary = summary
        conv.summarized_count = summarized_count


async def fold_summary(
    state: HistoryState, llm: LLMClient, settings: Settings
) -> str | None:
    """Fold aged-out messages into the summary. None means it failed and nothing advanced."""
    transcript = "\n".join(f"{r.title()}: {t}" for r, t in state.to_fold)
    user = f"Previous summary:\n{state.summary or '(none)'}\n\nNew messages:\n{transcript}"
    try:
        resp = await llm.complete(
            LLMRequest(
                system=SUMMARY_PROMPT_V1,
                user=user,
                model=settings.pipeline.rewrite_model,
                max_tokens=400,
            )
        )
    except Exception:
        log.exception("summarization failed")
        return None
    return resp.text.strip() or None


def history_text(summary: str, recent: list[tuple[str, str]]) -> str:
    return render_history(summary, recent)
