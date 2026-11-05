import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Any

from app.config import PipelineConfig, Settings
from app.services.generation.llm_client import LLMClient, LLMRequest
from app.services.generation.prompts import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    build_user_message,
    get_system_prompt,
    render_document_block,
    render_source,
)
from app.services.ingestion.types import count_tokens
from app.services.turn import Group, Turn
from app.services.vectorstore import Hit

_MARKER = re.compile(r"\[(c_\d+)\]")
QUOTE_CHARS = 300


@dataclass
class Citation:
    marker: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    page_start: int
    page_end: int
    quote: str = ""
    bbox_list: list[Any] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["chunk_id"], d["document_id"] = str(self.chunk_id), str(self.document_id)
        return d


@dataclass
class RagAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    closest: list[dict[str, Any]] = field(default_factory=list)  # passages shown when gated
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class StreamEvent:
    """One SSE event. `event` is the SSE event name, `data` is JSON-serializable."""

    event: str
    data: dict[str, Any]


def closest_json(hits: list[Hit]) -> list[dict[str, Any]]:
    return [
        {
            "document_id": str(h.document_id),
            "document_title": h.document_title,
            "page_start": h.page_start,
            "quote": h.child_text[:QUOTE_CHARS],
            "bbox_list": h.bbox_list,
            "score": round(h.score, 3),
        }
        for h in hits
    ]


def _pack_group(
    group: Group, budget: int, by_marker: dict[str, Hit], seen: set[uuid.UUID]
) -> list[str]:
    """Expand children to parents, dedupe, and stop at the token budget (not at a fixed k)."""
    rendered: list[str] = []
    used = 0
    for hit in group.hits:
        if hit.chunk_id in seen:
            continue
        cost = count_tokens(hit.parent_text)
        if rendered and used + cost > budget:
            break
        seen.add(hit.chunk_id)
        used += cost
        marker = f"c_{len(by_marker) + 1}"
        by_marker[marker] = hit
        rendered.append(render_source(marker, hit.document_title, hit.page_start, hit.parent_text))
    return rendered


def pack_context(
    groups: list[Group], budget: int, grouped: bool = False
) -> tuple[list[str], dict[str, Hit]]:
    """Return rendered source blocks and the marker -> hit map used to validate citations."""
    by_marker: dict[str, Hit] = {}
    seen: set[uuid.UUID] = set()
    per_group = max(1, budget // max(1, len(groups)))
    blocks: list[str] = []
    for g in groups:
        rendered = _pack_group(g, per_group, by_marker, seen)
        blocks.extend([render_document_block(g.title or "", rendered)] if grouped else rendered)
    return blocks, by_marker


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
                marker,
                hit.chunk_id,
                hit.document_id,
                hit.document_title,
                hit.page_start,
                hit.page_end,
                quote=hit.child_text,
                bbox_list=hit.bbox_list,
            ),
        )
        return match.group(0)

    cleaned = _MARKER.sub(keep, answer)
    return re.sub(r"[ \t]+([.,;])", r"\1", cleaned), list(cited.values())


def _prepare(
    turn: Turn, settings: Settings, config: PipelineConfig
) -> tuple[LLMRequest, dict[str, Hit]]:
    grouped = turn.mode == "compare"
    rendered, by_marker = pack_context(turn.groups, config.context_token_budget, grouped)
    request = LLMRequest(
        system=get_system_prompt(turn.mode, config.prompt_version),
        user=build_user_message(turn.question, rendered, turn.history_text),
        model=settings.llm_model,
    )
    return request, by_marker


def _refusal(turn: Turn) -> RagAnswer:
    return RagAnswer(INSUFFICIENT_CONTEXT_MESSAGE, closest=closest_json(turn.closest))


def _has_context(turn: Turn) -> bool:
    return not turn.gated and any(g.hits for g in turn.groups)


async def answer_question(
    turn: Turn, llm: LLMClient, settings: Settings, config: PipelineConfig
) -> RagAnswer:
    if not _has_context(turn):
        return _refusal(turn)  # no LLM call: nothing relevant to ground an answer in
    request, by_marker = _prepare(turn, settings, config)
    resp = await llm.complete(request)
    text, citations = validate_citations(resp.text, by_marker)
    return RagAnswer(text, citations, [], resp.input_tokens, resp.output_tokens)


async def stream_answer(
    turn: Turn, llm: LLMClient, settings: Settings, config: PipelineConfig
) -> AsyncIterator[StreamEvent | RagAnswer]:
    """Yield token and citation events, then the final RagAnswer (not an SSE event itself).

    Raw tokens may contain markers that are later dropped; the final answer is authoritative.
    """
    if not _has_context(turn):
        refusal = _refusal(turn)
        yield StreamEvent("token", {"text": refusal.answer})
        if refusal.closest:
            yield StreamEvent("closest", {"passages": refusal.closest})
        yield refusal
        return
    request, by_marker = _prepare(turn, settings, config)
    yield StreamEvent("status", {"stage": "writing"})
    raw: list[str] = []
    in_tokens = out_tokens = 0
    async for ev in llm.stream(request):
        if ev.kind == "delta":
            raw.append(ev.text)
            yield StreamEvent("token", {"text": ev.text})
        elif ev.kind == "usage":
            in_tokens, out_tokens = ev.input_tokens, ev.output_tokens
    text, citations = validate_citations("".join(raw), by_marker)
    for c in citations:
        yield StreamEvent("citation", c.to_json())
    yield RagAnswer(text, citations, [], in_tokens, out_tokens)
