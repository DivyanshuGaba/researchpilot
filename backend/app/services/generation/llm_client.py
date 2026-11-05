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


class LLMClient(Protocol):
    async def complete(self, req: LLMRequest) -> LLMResponse: ...


class AnthropicClient:
    """Thin wrapper. The SDK retries 429/5xx with backoff; we set the retry count and timeout."""

    def __init__(self, api_key: str, max_retries: int = 3, timeout: float = 60.0) -> None:
        self._client = AsyncAnthropic(api_key=api_key, max_retries=max_retries, timeout=timeout)

    async def complete(self, req: LLMRequest) -> LLMResponse:
        msg = await self._client.messages.create(
            model=req.model,
            max_tokens=req.max_tokens,
            # Static system prompt first and marked cacheable (architecture.md 5.6).
            system=[{"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": req.user}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResponse(text, msg.usage.input_tokens, msg.usage.output_tokens)
