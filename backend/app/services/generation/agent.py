"""Bounded tool-use loop (architecture.md section 6).

Hard limits: tool rounds, total tool calls, a token budget and a wall-clock timeout. When any
limit is hit the model is asked, without tools, to answer with what it has, and the answer says
it stopped early. Every tool call is logged and streamed to the client.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.services.generation.llm_client import ConverseRequest, LLMClient, ToolCall
from app.services.generation.prompts import (
    WRAP_UP_NUDGE,
    empty_answer_message,
    get_system_prompt,
)
from app.services.generation.rag_chain import Citation, StreamEvent, validate_citations
from app.services.generation.tools import ToolBox, ToolResult
from app.services.vectorstore import Hit

log = logging.getLogger(__name__)

Emit = Callable[[StreamEvent], Awaitable[None]]


@dataclass(frozen=True)
class AgentLimits:
    max_iterations: int = 5
    max_tool_calls: int = 8
    max_total_tokens: int = 60_000
    timeout_s: float = 60.0
    wrapup_timeout_s: float = 25.0
    tool_timeout_s: float = 30.0


@dataclass
class AgentResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    stopped: str | None = None  # max_iterations | max_tool_calls | token_budget | timeout
    input_tokens: int = 0
    output_tokens: int = 0


STOP_NOTES = {
    "max_iterations": "reached the limit of tool rounds",
    "max_tool_calls": "reached the limit of tool calls",
    "token_budget": "reached the token budget",
    "timeout": "ran out of time",
}


def _preview(args: dict[str, Any]) -> dict[str, Any]:
    return {k: (v[:200] if isinstance(v, str) else v) for k, v in args.items()}


def _append_user_text(messages: list[dict[str, Any]], text: str) -> None:
    """Add text to the trailing user turn (the API rejects two user turns in a row)."""
    block = {"type": "text", "text": text}
    if messages and messages[-1]["role"] == "user":
        content = messages[-1]["content"]
        messages[-1]["content"] = (
            [{"type": "text", "text": content}, block] if isinstance(content, str) else [*content, block]
        )
    else:
        messages.append({"role": "user", "content": [block]})


async def run_agent(
    *,
    question: str,
    history_text: str,
    toolbox: ToolBox,
    by_marker: dict[str, Hit],
    llm: LLMClient,
    settings: Settings,
    limits: AgentLimits = AgentLimits(),  # noqa: B008 - frozen, safe as a default
    emit: Emit,
) -> AgentResult:
    system = get_system_prompt("agent", settings.pipeline.prompt_version)
    first = f"{history_text}\n\nQuestion: {question}" if history_text else question
    messages: list[dict[str, Any]] = [{"role": "user", "content": first}]
    result = AgentResult(answer="")
    started = time.monotonic()
    final: str | None = None
    calls_made = 0

    async def converse(tools: bool, timeout: float, wrap_up: bool = False) -> Any:
        return await asyncio.wait_for(
            llm.converse(
                ConverseRequest(
                    system=system,
                    messages=messages,
                    tools=toolbox.schemas if tools or wrap_up else [],
                    model=settings.llm_model,
                    tool_choice={"type": "none"} if wrap_up else None,
                )
            ),
            timeout,
        )

    try:
        for _ in range(limits.max_iterations):
            remaining = limits.timeout_s - (time.monotonic() - started)
            if remaining <= 0:
                result.stopped = "timeout"
                break
            turn = await converse(True, remaining)
            result.input_tokens += turn.input_tokens
            result.output_tokens += turn.output_tokens
            if not turn.tool_calls:
                final = turn.text.strip() or empty_answer_message(turn.stop_reason)
                break
            messages.append({"role": "assistant", "content": turn.content_blocks})
            results: list[dict[str, Any]] = []
            for call in turn.tool_calls:
                if calls_made >= limits.max_tool_calls:
                    out = ToolResult("Error: tool call limit reached", is_error=True)
                    result.stopped = result.stopped or "max_tool_calls"
                else:
                    calls_made += 1
                    out = await _run_tool(call, toolbox, limits, result, emit)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": out.output,
                        "is_error": out.is_error,
                    }
                )
            messages.append({"role": "user", "content": results})
            if result.input_tokens + result.output_tokens > limits.max_total_tokens:
                result.stopped = "token_budget"
                break
            if result.stopped == "max_tool_calls":
                break
        else:
            result.stopped = "max_iterations"
    except TimeoutError:
        result.stopped = "timeout"

    if final is None:  # a limit was hit: answer with what we have, without tools
        _append_user_text(messages, WRAP_UP_NUDGE)
        turn = await converse(False, limits.wrapup_timeout_s, wrap_up=True)
        result.input_tokens += turn.input_tokens
        result.output_tokens += turn.output_tokens
        note = STOP_NOTES.get(result.stopped or "", "stopped early")
        body = turn.text.strip() or empty_answer_message(turn.stop_reason)
        final = f"{body}\n\n(Stopped early: {note}, so this answer may be incomplete.)"

    result.answer, result.citations = validate_citations(final, by_marker)
    log.info(
        "agent done: tools=%d stopped=%s tokens=%d",
        len(result.tool_log), result.stopped, result.input_tokens + result.output_tokens,
    )
    return result


async def _run_tool(
    call: ToolCall, toolbox: ToolBox, limits: AgentLimits, result: AgentResult, emit: Emit
) -> ToolResult:
    base = {"id": call.id, "name": call.name, "input": _preview(call.input)}
    await emit(StreamEvent("tool", {**base, "status": "started"}))
    t0 = time.perf_counter()
    try:
        out = await asyncio.wait_for(toolbox.run(call.name, call.input), limits.tool_timeout_s)
    except TimeoutError:
        out = ToolResult("Error: the tool timed out", is_error=True)
    ms = int((time.perf_counter() - t0) * 1000)
    record = {**base, "status": "finished", "duration_ms": ms,
              "output_chars": len(out.output), "is_error": out.is_error}  # fmt: skip
    result.tool_log.append(record)
    log.info(
        "agent tool call: name=%s input=%s ms=%d out_chars=%d error=%s",
        call.name, base["input"], ms, len(out.output), out.is_error,
    )
    await emit(StreamEvent("tool", record))
    return out
