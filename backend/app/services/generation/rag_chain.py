import re
import uuid
from dataclasses import dataclass, field

from app.config import PipelineConfig, Settings
from app.services.generation.llm_client import LLMClient, LLMRequest
from app.services.generation.prompts import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    PROMPTS,
    build_user_message,
    render_source,
)
from app.services.vectorstore import Hit

_MARKER = re.compile(r"\[(c_\d+)\]")


@dataclass
class Citation:
    marker: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    page_start: int
    page_end: int


@dataclass
class RagAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def pack_context(hits: list[Hit]) -> tuple[list[str], dict[str, Hit]]:
    """Expand children to parents, dedupe, and assign short ids the model can cite."""
    seen: set[uuid.UUID] = set()
    rendered: list[str] = []
    by_marker: dict[str, Hit] = {}
    for hit in hits:
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        marker = f"c_{len(by_marker) + 1}"
        by_marker[marker] = hit
        rendered.append(render_source(marker, hit.document_title, hit.page_start, hit.parent_text))
    return rendered, by_marker


def validate_citations(answer: str, by_marker: dict[str, Hit]) -> tuple[str, list[Citation]]:
    """The model proposes, the server verifies: drop markers that are not in the packed context."""
    cited: dict[str, Citation] = {}

    def keep(match: re.Match[str]) -> str:
        marker = match.group(1)
        hit = by_marker.get(marker)
        if hit is None:
            return ""
        cited.setdefault(
            marker,
            Citation(
                marker, hit.chunk_id, hit.document_id, hit.document_title,
                hit.page_start, hit.page_end,
            ),
        )
        return match.group(0)

    cleaned = _MARKER.sub(keep, answer)
    return re.sub(r"[ \t]+([.,;])", r"\1", cleaned), list(cited.values())


async def answer_question(
    question: str,
    hits: list[Hit],
    llm: LLMClient,
    settings: Settings,
    config: PipelineConfig,
) -> RagAnswer:
    if not hits:
        return RagAnswer(INSUFFICIENT_CONTEXT_MESSAGE)
    rendered, by_marker = pack_context(hits)
    resp = await llm.complete(
        LLMRequest(
            system=PROMPTS[config.prompt_version],
            user=build_user_message(question, rendered),
            model=settings.llm_model,
        )
    )
    text, citations = validate_citations(resp.text, by_marker)
    return RagAnswer(text, citations, resp.input_tokens, resp.output_tokens)
