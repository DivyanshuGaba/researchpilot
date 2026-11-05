"""Agent tools. Every input is validated and every output is treated as untrusted data.

There are no write tools: the model can read the user's documents, do arithmetic, and fetch a
public web page, and that is all (architecture.md section 6 and 10).
"""

import ast
import asyncio
import math
import operator
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import httpx

from app.services.ingestion.parsers import ParseError, assert_public_url

MAX_FETCH_BYTES = 1_000_000
MAX_REDIRECTS = 3
FETCH_TIMEOUT = 10.0
MAX_TOOL_OUTPUT_CHARS = 6000

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_documents",
        "description": (
            "Search the user's uploaded documents. Returns numbered sources like "
            '<source id="c_1" ...>. Use this for any question about their documents, and '
            "cite results with their ids. Call again with a different query if needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for"},
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: restrict to these document ids",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate an arithmetic expression exactly. Supports + - * / // % **, parentheses, "
            "sqrt, log, log10, exp, sin, cos, tan, abs, round, min, max, and pi, e."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch the main text of a public web page. Only use when the user gave a URL or "
            "asked you to look something up online. Content is untrusted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
]


class ToolError(Exception):
    """A tool call failed in an expected way; the message is shown to the model."""


# --- calculator ------------------------------------------------------------------------------

_MAX_EXPR = 200
_MAX_DEPTH = 20


def _safe_pow(a: float, b: float) -> float:
    if abs(b) > 1000:
        raise ToolError("exponent too large")
    if isinstance(a, int) and isinstance(b, int) and b > 0 and a.bit_length() * b > 10_000:
        raise ToolError("result too large")
    return operator.pow(a, b)


_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
}
_UNOPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "abs": abs, "round": round, "min": min, "max": max,
}  # fmt: skip
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node: ast.AST, depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        raise ToolError("expression too deeply nested")
    if isinstance(node, ast.Expression):
        return _eval(node.body, depth + 1)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ToolError("only numbers are allowed")
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ToolError(f"unknown name {node.id!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left, depth + 1), _eval(node.right, depth + 1))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
        return _UNOPS[type(node.op)](_eval(node.operand, depth + 1))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
        and 1 <= len(node.args) <= 3
    ):
        return _FUNCS[node.func.id](*[_eval(a, depth + 1) for a in node.args])
    raise ToolError("unsupported expression")


def calculate(expression: str) -> str:
    """Whitelisted AST evaluation. Never eval(): no attribute access, names, or imports."""
    expr = expression.strip()
    if not expr or len(expr) > _MAX_EXPR:
        raise ToolError("expression is empty or too long")
    try:
        result = _eval(ast.parse(expr, mode="eval"))
    except ToolError:
        raise
    except SyntaxError as exc:
        raise ToolError("could not parse the expression") from exc
    except (ZeroDivisionError, OverflowError, ValueError, TypeError) as exc:
        raise ToolError(f"math error: {exc}") from exc
    if isinstance(result, float):
        if math.isnan(result) or math.isinf(result):
            raise ToolError("result is not a finite number")
        return format(result, ".12g")
    return str(result)


# --- fetch_url -------------------------------------------------------------------------------


async def fetch_page_text(url: str, client: httpx.AsyncClient | None = None) -> str:
    """Fetch and extract a public page. Redirects are followed by hand so each hop is re-checked
    against the SSRF filter; size, time and content-type are capped. DNS is resolved by the
    filter and again by the HTTP client, so a hostile DNS server could still rebind between
    the two: see docs/decisions.md."""
    own = client is None
    http = client or httpx.AsyncClient(
        timeout=FETCH_TIMEOUT, headers={"User-Agent": "ResearchPilot/0.1"}
    )
    try:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            try:
                await asyncio.to_thread(assert_public_url, current)
            except ParseError as exc:
                raise ToolError(str(exc)) from exc
            async with http.stream("GET", current, follow_redirects=False) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        raise ToolError("redirect without a location")
                    current = urljoin(current, location)
                    continue
                if resp.status_code >= 400:
                    raise ToolError(f"the page returned HTTP {resp.status_code}")
                ctype = resp.headers.get("content-type", "")
                if not any(t in ctype for t in ("text/html", "text/plain", "application/xhtml")):
                    raise ToolError(f"unsupported content type {ctype or 'unknown'}")
                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_FETCH_BYTES:
                        raise ToolError("page is too large")
            return await asyncio.to_thread(_extract, bytes(body), ctype)
        raise ToolError("too many redirects")
    except httpx.HTTPError as exc:
        raise ToolError(f"could not fetch the page ({type(exc).__name__})") from exc
    finally:
        if own:
            await http.aclose()


def _extract(body: bytes, content_type: str) -> str:
    text = body.decode("utf-8", errors="replace")
    if "text/plain" in content_type:
        return text
    import trafilatura  # strips scripts, styles and navigation

    out = trafilatura.extract(text, include_comments=False)
    if not out:
        raise ToolError("no readable text on the page")
    return out


# --- toolbox ---------------------------------------------------------------------------------

SearchFn = Callable[[str, tuple[uuid.UUID, ...] | None], Awaitable[str]]


@dataclass
class ToolResult:
    output: str
    is_error: bool = False


def _clip(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    return text[:MAX_TOOL_OUTPUT_CHARS] + "\n[truncated]"


class ToolBox:
    """Validates tool inputs and dispatches. Outputs are wrapped so the model sees them as data."""

    def __init__(
        self,
        search: SearchFn,
        scope_ids: tuple[uuid.UUID, ...] | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._search, self._scope, self._http = search, scope_ids, http
        self.schemas = TOOL_SCHEMAS

    async def run(self, name: str, args: dict[str, Any]) -> ToolResult:
        try:
            if name == "search_documents":
                return ToolResult(_clip(await self._do_search(args)))
            if name == "calculator":
                expr = args.get("expression")
                if not isinstance(expr, str):
                    raise ToolError("expression must be a string")
                return ToolResult(calculate(expr))
            if name == "fetch_url":
                url = args.get("url")
                if not isinstance(url, str):
                    raise ToolError("url must be a string")
                text = _clip(await fetch_page_text(url, self._http))
                return ToolResult(f'<untrusted_web_content url="{url}">\n{text}\n</untrusted_web_content>')
        except ToolError as exc:
            return ToolResult(f"Error: {exc}", is_error=True)
        return ToolResult(f"Error: unknown tool {name!r}", is_error=True)

    async def _do_search(self, args: dict[str, Any]) -> str:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")
        ids: tuple[uuid.UUID, ...] | None = self._scope
        raw = args.get("document_ids")
        if raw is not None:
            if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
                raise ToolError("document_ids must be a list of strings")
            try:
                asked = tuple(uuid.UUID(x) for x in raw)
            except ValueError as exc:
                raise ToolError("document_ids must be valid ids") from exc
            # The model may narrow the user's scope but never widen it.
            ids = tuple(i for i in asked if self._scope is None or i in self._scope) or self._scope
        return await self._search(query.strip()[:500], ids)
