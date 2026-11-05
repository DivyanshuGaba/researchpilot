import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic
from anthropic import AsyncAnthropic

from app.config import PRICES_PER_MTOK

log = logging.getLogger(__name__)

SAFE_MESSAGES = {
    "rate_limited": "The model is busy right now. Please try again in a moment.",
    "timeout": "The model took too long to respond. Please try again.",
    "unavailable": "The model service is temporarily unavailable. Please try again shortly.",
    "misconfigured": "The model service is not configured correctly. Please contact the administrator.",
    "billing": "The model service is unavailable because of an account issue. Please contact the administrator.",
}


class LLMUnavailable(Exception):
    """The model call failed for a reason the caller can describe safely to a user. The message
    never contains provider internals (request ids, keys, raw error bodies)."""

    def __init__(self, kind: str, retry_after: int | None = None) -> None:
        super().__init__(kind)
        self.kind = kind
        self.retry_after = retry_after
        self.message = SAFE_MESSAGES.get(kind, SAFE_MESSAGES["unavailable"])


def classify_error(exc: Exception) -> LLMUnavailable | None:
    """Map an SDK exception to a safe category. None means it is not a provider failure."""
    if isinstance(exc, anthropic.RateLimitError):
        return LLMUnavailable("rate_limited", retry_after=30)
    if isinstance(exc, anthropic.APITimeoutError | TimeoutError):
        return LLMUnavailable("timeout")
    if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        log.error("model provider rejected our credentials")  # operator problem, not the user's
        return LLMUnavailable("misconfigured")
    if isinstance(exc, anthropic.BadRequestError) and "credit balance" in str(exc).lower():
        log.error("model provider reports the credit balance is too low")  # operator problem
        return LLMUnavailable("billing")
    if isinstance(exc, anthropic.APIConnectionError | anthropic.InternalServerError):
        return LLMUnavailable("unavailable")
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
        return LLMUnavailable("unavailable")
    return None


@dataclass
class LLMRequest:
    system: str
    user: str
    model: str
    max_tokens: int = 1024


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str = ""
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class LLMEvent:
    """kind is "delta" (text set) or "usage" (token counts set, emitted once at the end)."""

    kind: str
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ConverseRequest:
    """A multi-message request that may offer tools (used by the agent loop)."""

    system: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    model: str
    max_tokens: int = 1500
    tool_choice: dict[str, Any] | None = None  # {"type": "none"} forces a text-only reply


@dataclass
class ToolTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    content_blocks: list[dict[str, Any]] = field(default_factory=list)  # replay as assistant turn
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class LLMClient(Protocol):
    async def complete(self, req: LLMRequest) -> LLMResponse: ...
    def stream(self, req: LLMRequest) -> AsyncIterator[LLMEvent]: ...
    async def converse(self, req: ConverseRequest) -> ToolTurn: ...


class AnthropicClient:
    """Thin wrapper. The SDK retries 429/5xx with backoff; we set the retry count and timeout."""

    def __init__(
        self, api_key: str, max_retries: int = 3, timeout: float = 60.0, max_concurrency: int = 8
    ) -> None:
        self._client = AsyncAnthropic(api_key=api_key, max_retries=max_retries, timeout=timeout)
        # One shared ceiling on in-flight model calls, so a burst cannot exhaust the rate limit.
        self._sem = asyncio.Semaphore(max_concurrency)

    def _system(self, req: LLMRequest) -> list[dict[str, object]]:
        # Static system prompt first and marked cacheable (architecture.md 5.6).
        return [{"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}]

    async def stream(self, req: LLMRequest) -> AsyncIterator[LLMEvent]:
        try:
            async with (
                self._sem,
                self._client.messages.stream(
                    model=req.model,
                    max_tokens=req.max_tokens,
                    system=self._system(req),  # type: ignore[arg-type]
                    messages=[{"role": "user", "content": req.user}],
                ) as stream,
            ):
                async for delta in stream.text_stream:
                    yield LLMEvent("delta", text=delta)
                final = await stream.get_final_message()
        except Exception as exc:
            mapped = classify_error(exc)
            if mapped is None:
                raise
            raise mapped from exc
        yield LLMEvent(
            "usage",
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            stop_reason=final.stop_reason or "",
            cache_read_tokens=getattr(final.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(final.usage, "cache_creation_input_tokens", 0) or 0,
        )

    async def complete(self, req: LLMRequest) -> LLMResponse:
        try:
            async with self._sem:
                msg = await self._client.messages.create(
                    model=req.model,
                    max_tokens=req.max_tokens,
                    system=self._system(req),  # type: ignore[arg-type]
                    messages=[{"role": "user", "content": req.user}],
                )
        except Exception as exc:
            mapped = classify_error(exc)
            if mapped is None:
                raise
            raise mapped from exc
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResponse(
            text, msg.usage.input_tokens, msg.usage.output_tokens, msg.stop_reason or "",
            getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
            getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
        )

    async def converse(self, req: ConverseRequest) -> ToolTurn:
        extra: dict[str, Any] = {"tools": req.tools} if req.tools else {}
        if req.tool_choice and req.tools:
            extra["tool_choice"] = req.tool_choice
        try:
            async with self._sem:
                msg = await self._client.messages.create(
                    model=req.model,
                    max_tokens=req.max_tokens,
                    system=self._system(req),  # type: ignore[arg-type]
                    messages=req.messages,  # type: ignore[arg-type]
                    **extra,
                )
        except Exception as exc:
            mapped = classify_error(exc)
            if mapped is None:
                raise
            raise mapped from exc
        blocks: list[dict[str, Any]] = []
        calls: list[ToolCall] = []
        text_parts: list[str] = []
        for b in msg.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
                text_parts.append(b.text)
            elif b.type == "tool_use":
                inp = dict(b.input) if isinstance(b.input, dict) else {}
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": inp})
                calls.append(ToolCall(b.id, b.name, inp))
        return ToolTurn(
            "".join(text_parts), calls, blocks, msg.usage.input_tokens, msg.usage.output_tokens,
            msg.stop_reason or "",
            getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
            getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
        )


@dataclass
class Usage:
    """Token and cost totals across every model call made for one request."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


def cost_usd(model: str, inp: int, out: int, cache_read: int = 0, cache_write: int = 0) -> float:
    """Cache reads bill at 0.1x the input price and cache writes at 1.25x (5 minute TTL)."""
    price = PRICES_PER_MTOK.get(model)
    if price is None:
        return 0.0
    pin, pout = price
    return (inp * pin + out * pout + cache_read * pin * 0.1 + cache_write * pin * 1.25) / 1_000_000


class MeteredLLM:
    """Wraps any LLMClient and totals usage and cost, so a request knows what it spent."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.usage = Usage()

    def _add(self, model: str, r: Any) -> None:
        u = self.usage
        u.input_tokens += r.input_tokens
        u.output_tokens += r.output_tokens
        u.cache_read_tokens += r.cache_read_tokens
        u.cache_write_tokens += r.cache_write_tokens
        u.cost_usd += cost_usd(
            model, r.input_tokens, r.output_tokens, r.cache_read_tokens, r.cache_write_tokens
        )
        u.calls += 1

    async def complete(self, req: LLMRequest) -> LLMResponse:
        resp = await self.inner.complete(req)
        self._add(req.model, resp)
        return resp

    async def converse(self, req: ConverseRequest) -> ToolTurn:
        turn = await self.inner.converse(req)
        self._add(req.model, turn)
        return turn

    async def stream(self, req: LLMRequest) -> AsyncIterator[LLMEvent]:
        async for ev in self.inner.stream(req):
            if ev.kind == "usage":
                self._add(req.model, ev)
            yield ev
