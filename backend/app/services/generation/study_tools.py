"""Quiz, flashcard and summary generation (architecture.md section 5.7).

Passages are sampled evenly across the document's sections (not randomly), the model must return
strict JSON that Pydantic validates, and an invalid reply is retried once with the validation
error appended. Every question and card keeps the id and pages of the passage it came from.
Summaries are map-reduce: each section is summarized independently (cheap model, in parallel),
then combined; section summaries are cached per document version.
"""

import asyncio
import json
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.schemas.study import CardsDraft, QuizDraft
from app.services.generation.llm_client import LLMClient, LLMRequest
from app.services.generation.prompts import (
    FLASHCARD_PROMPT_V1,
    QUIZ_PROMPT_V1,
    REDUCE_SUMMARY_PROMPT_V1,
    SECTION_SUMMARY_PROMPT_V1,
)
from app.services.ingestion.types import count_tokens

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class StudyGenerationError(Exception):
    """The model could not produce a valid result (shown to the user as a safe message)."""


@dataclass(frozen=True)
class Section:
    chunk_id: uuid.UUID
    page_start: int
    page_end: int
    text: str


# --- sampling --------------------------------------------------------------------------------


def stratified_sample(sections: list[Section], n: int) -> list[Section]:
    """Pick n sections spread evenly across the document: one from the middle of each of n equal
    bins, so the whole document is covered and the result is deterministic."""
    if n <= 0 or not sections:
        return []
    if n >= len(sections):
        return list(sections)
    size = len(sections) / n
    return [sections[min(len(sections) - 1, int(i * size + size / 2))] for i in range(n)]


def render_passages(passages: list[Section]) -> str:
    out = []
    for i, p in enumerate(passages, start=1):
        pages = str(p.page_start) if p.page_start == p.page_end else f"{p.page_start}-{p.page_end}"
        out.append(f'<passage id="s_{i}" pages="{pages}">\n{p.text}\n</passage>')
    return "\n".join(out)


# --- validated JSON generation ---------------------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> object:
    """Tolerate code fences and stray prose around the object; still fail on bad JSON."""
    t = _FENCE.sub("", text.strip())
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(t[start : end + 1])


async def generate_validated(
    llm: LLMClient,
    *,
    system: str,
    user: str,
    model: str,
    schema: type[T],
    check: Callable[[T], None] | None = None,
    max_tokens: int = 4000,
    retries: int = 1,
) -> tuple[T, int]:
    """Returns (validated result, attempts used). Retries once with the error appended."""
    prompt = user
    last_error = ""
    for attempt in range(retries + 1):
        resp = await llm.complete(
            LLMRequest(system=system, user=prompt, model=model, max_tokens=max_tokens)
        )
        try:
            result = schema.model_validate(extract_json(resp.text))
            if check:
                check(result)
            return result, attempt + 1
        except (ValueError, ValidationError) as exc:  # JSONDecodeError and pydantic are ValueErrors
            last_error = _short_error(exc)
            log.warning("study generation invalid (attempt %d): %s", attempt + 1, last_error)
            prompt = (
                f"{user}\n\nYour previous reply was invalid: {last_error}\n"
                "Reply again with ONLY a corrected JSON object that follows every rule."
            )
    raise StudyGenerationError(f"the model returned invalid output twice: {last_error}")


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]]
        return "; ".join(parts)
    return str(exc)[:300]


def _source_checker(passage_count: int) -> Callable[[BaseModel], None]:
    valid = {f"s_{i}" for i in range(1, passage_count + 1)}

    def check(result: BaseModel) -> None:
        items = getattr(result, "questions", None) or getattr(result, "cards", [])
        bad = sorted({i.source for i in items if i.source not in valid})
        if bad:
            raise ValueError(f"unknown source ids {bad}; use only {sorted(valid)}")

    return check


# --- quiz and flashcards ---------------------------------------------------------------------


@dataclass
class GeneratedQuiz:
    questions: list[dict[str, object]]
    requested: int
    attempts: int


async def generate_quiz(
    llm: LLMClient, settings: Settings, sections: list[Section], num_questions: int
) -> GeneratedQuiz:
    passages = stratified_sample(sections, min(settings.pipeline.quiz_max_passages, num_questions))
    user = f"Write {num_questions} questions.\n\n{render_passages(passages)}"
    draft, attempts = await generate_validated(
        llm, system=QUIZ_PROMPT_V1, user=user, model=settings.llm_model, schema=QuizDraft,
        check=_source_checker(len(passages)),
    )
    questions: list[dict[str, object]] = []
    for q in draft.questions[:num_questions]:
        src = passages[int(q.source[2:]) - 1]
        questions.append(
            {
                "question": q.question,
                "options": q.options,
                "answer_index": q.answer_index,
                "explanation": q.explanation,
                "chunk_id": str(src.chunk_id),
                "page_start": src.page_start,
                "page_end": src.page_end,
            }
        )
    return GeneratedQuiz(questions, num_questions, attempts)


@dataclass
class GeneratedCards:
    cards: list[dict[str, object]]
    requested: int
    attempts: int


async def generate_flashcards(
    llm: LLMClient, settings: Settings, sections: list[Section], num_cards: int
) -> GeneratedCards:
    passages = stratified_sample(sections, min(settings.pipeline.quiz_max_passages, num_cards))
    user = f"Write {num_cards} flashcards.\n\n{render_passages(passages)}"
    draft, attempts = await generate_validated(
        llm, system=FLASHCARD_PROMPT_V1, user=user, model=settings.llm_model, schema=CardsDraft,
        check=_source_checker(len(passages)),
    )
    cards: list[dict[str, object]] = []
    for c in draft.cards[:num_cards]:
        src = passages[int(c.source[2:]) - 1]
        cards.append(
            {
                "front": c.front,
                "back": c.back,
                "chunk_id": str(src.chunk_id),
                "page_start": src.page_start,
                "page_end": src.page_end,
            }
        )
    return GeneratedCards(cards, num_cards, attempts)


# --- map-reduce summary ----------------------------------------------------------------------


def _pages(s: Section) -> str:
    return f"{s.page_start}" if s.page_start == s.page_end else f"{s.page_start}-{s.page_end}"


async def summarize_sections(
    llm: LLMClient, settings: Settings, sections: list[Section], cached: dict[uuid.UUID, str]
) -> dict[uuid.UUID, str]:
    """Map step: one cheap call per section, bounded parallelism. Cached sections are reused."""
    sem = asyncio.Semaphore(settings.pipeline.summary_concurrency)
    out = dict(cached)

    async def one(sec: Section) -> None:
        async with sem:
            resp = await llm.complete(
                LLMRequest(
                    system=SECTION_SUMMARY_PROMPT_V1,
                    user=f"<passage pages=\"{_pages(sec)}\">\n{sec.text}\n</passage>",
                    model=settings.pipeline.rewrite_model,
                    max_tokens=300,
                )
            )
        out[sec.chunk_id] = resp.text.strip()

    await asyncio.gather(*(one(s) for s in sections if s.chunk_id not in cached))
    return out


async def reduce_summaries(
    llm: LLMClient, settings: Settings, lines: list[str], depth: int = 0
) -> str:
    """Reduce step. If the section summaries do not fit one call, reduce them in groups first."""
    budget = settings.pipeline.summary_reduce_tokens
    total = sum(count_tokens(x) for x in lines)
    if total > budget and len(lines) > 1 and depth < 4:
        groups: list[list[str]] = [[]]
        size = 0
        for line in lines:
            t = count_tokens(line)
            if groups[-1] and size + t > budget:
                groups.append([])
                size = 0
            groups[-1].append(line)
            size += t
        partial = await asyncio.gather(
            *(reduce_summaries(llm, settings, g, depth + 1) for g in groups)
        )
        return await reduce_summaries(llm, settings, [f"[group {i + 1}] {p}" for i, p in enumerate(partial)], depth + 1)
    resp = await llm.complete(
        LLMRequest(
            system=REDUCE_SUMMARY_PROMPT_V1,
            user="\n".join(lines),
            model=settings.llm_model,
            max_tokens=900,
        )
    )
    return resp.text.strip()


async def summarize_document(
    llm: LLMClient, settings: Settings, sections: list[Section], cached: dict[uuid.UUID, str]
) -> tuple[str, dict[uuid.UUID, str], bool]:
    """Returns (summary, section summaries for caching, truncated)."""
    cap = settings.pipeline.summary_max_sections
    truncated = len(sections) > cap
    used = stratified_sample(sections, cap) if truncated else sections
    summaries = await summarize_sections(llm, settings, used, cached)
    lines = [f"[p.{_pages(s)}] {summaries[s.chunk_id]}" for s in used]
    return await reduce_summaries(llm, settings, lines), summaries, truncated
