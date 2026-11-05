from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from anthropic import AsyncAnthropic


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


@dataclass
class LLMEvent:
    """kind is "delta" (text set) or "usage" (token counts set, emitted once at the end)."""

    kind: str
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    async def complete(self, req: LLMRequest) -> LLMResponse: ...
    def stream(self, req: LLMRequest) -> AsyncIterator[LLMEvent]: ...


class AnthropicClient:
    """Thin wrapper. The SDK retries 429/5xx with backoff; we set the retry count and timeout."""

    def __init__(self, api_key: str, max_retries: int = 3, timeout: float = 60.0) -> None:
        self._client = AsyncAnthropic(api_key=api_key, max_retries=max_retries, timeout=timeout)

    def _system(self, req: LLMRequest) -> list[dict[str, object]]:
        # Static system prompt first and marked cacheable (architecture.md 5.6).
        return [{"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}]

    async def stream(self, req: LLMRequest) -> AsyncIterator[LLMEvent]:
        async with self._client.messages.stream(
            model=req.model,
            max_tokens=req.max_tokens,
            system=self._system(req),  # type: ignore[arg-type]
            messages=[{"role": "user", "content": req.user}],
        ) as stream:
            async for delta in stream.text_stream:
                yield LLMEvent("delta", text=delta)
            final = await stream.get_final_message()
        yield LLMEvent(
            "usage", input_tokens=final.usage.input_tokens, output_tokens=final.usage.output_tokens
        )

    async def complete(self, req: LLMRequest) -> LLMResponse:
        msg = await self._client.messages.create(
            model=req.model,
            max_tokens=req.max_tokens,
            system=self._system(req),  # type: ignore[arg-type]
            messages=[{"role": "user", "content": req.user}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResponse(text, msg.usage.input_tokens, msg.usage.output_tokens)
