"""Record real model replies once, replay them for free and deterministic CI runs."""

import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from app.services.generation.llm_client import (
    ConverseRequest,
    LLMEvent,
    LLMRequest,
    LLMResponse,
    ToolTurn,
)


class RecordingMiss(Exception):
    """A request was not in the recording: the prompt, chunks or retrieval changed."""


def request_key(req: LLMRequest) -> str:
    raw = json.dumps([req.system, req.user, req.model, req.max_tokens], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


class RecordingLLM:
    """Wraps a real client; every reply is saved to `path` (write on close)."""

    def __init__(self, inner: Any, path: Path) -> None:
        self.inner, self.path = inner, path
        self.store: dict[str, dict[str, Any]] = (
            json.loads(path.read_text()) if path.exists() else {}
        )

    async def complete(self, req: LLMRequest) -> LLMResponse:
        key = request_key(req)
        if key in self.store:  # re-recording only fills gaps
            d = self.store[key]
            return LLMResponse(d["text"], d["in"], d["out"], d.get("stop", ""), d.get("cr", 0), d.get("cw", 0))
        resp = await self.inner.complete(req)
        self.store[key] = {"text": resp.text, "in": resp.input_tokens, "out": resp.output_tokens,
                           "stop": resp.stop_reason, "cr": resp.cache_read_tokens, "cw": resp.cache_write_tokens}
        return resp

    def stream(self, req: LLMRequest) -> AsyncIterator[LLMEvent]:  # pragma: no cover
        raise NotImplementedError

    async def converse(self, req: ConverseRequest) -> ToolTurn:  # pragma: no cover
        raise NotImplementedError

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.store, indent=1, sort_keys=True))


class ReplayLLM:
    def __init__(self, path: Path) -> None:
        self.store: dict[str, dict[str, Any]] = json.loads(path.read_text())

    async def complete(self, req: LLMRequest) -> LLMResponse:
        d = self.store.get(request_key(req))
        if d is None:
            raise RecordingMiss(
                "request not in the recording (prompt, chunking or retrieval changed); "
                "re-record with: python -m eval.run_eval --smoke --record"
            )
        return LLMResponse(d["text"], d["in"], d["out"], d.get("stop", ""), d.get("cr", 0), d.get("cw", 0))

    def stream(self, req: LLMRequest) -> AsyncIterator[LLMEvent]:  # pragma: no cover
        raise NotImplementedError

    async def converse(self, req: ConverseRequest) -> ToolTurn:  # pragma: no cover
        raise NotImplementedError
