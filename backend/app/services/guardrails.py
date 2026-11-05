"""Defense in depth against prompt injection and data leakage (architecture.md section 10).

No single layer is trusted: scanning flags suspicious passages, the prompt marks them and
neutralizes our own delimiter tags inside document text, tools have no write access, and the
output sanitizer removes the channels an injected instruction would use to leak data.
"""

import logging
import re
import unicodedata
from urllib.parse import urlparse

# --- injection scanning -----------------------------------------------------------------------

# Zero-width and bidi-control code points, by number so no invisible characters live in this file.
_ZERO_WIDTH = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E]
)

INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_instructions": re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(previous|prior|above|earlier|preceding|all|any|your|the)\b"
        r"[^.\n]{0,30}\b(instructions?|prompts?|rules?|guidelines?|context|directions?)\b", re.IGNORECASE),
    "role_reassignment": re.compile(r"\b(you are now|from now on,? you|act as (if you were )?(an? )?(unrestricted|jailbroken|different)|pretend (to be|you are))\b", re.IGNORECASE),
    "prompt_extraction": re.compile(r"\b(reveal|show|print|repeat|output|leak)\b[^.\n]{0,30}\b(system prompt|your (initial |hidden )?instructions|your prompt)\b", re.IGNORECASE),
    "fake_system_message": re.compile(r"(^|\n)\s*(system|assistant|developer)\s*(message)?\s*:", re.IGNORECASE),
    "new_instructions": re.compile(r"\b(new|updated|additional|real) (instructions?|task|directive)s?\s*:", re.IGNORECASE),
    "delimiter_escape": re.compile(r"</?\s*(sources?|conversation|document|passage|untrusted_web_content)\b", re.IGNORECASE),
    "exfiltration": re.compile(
        r"!\[[^\]]*\]\(\s*https?://|\b(send|post|upload|forward|exfiltrate|transmit)\b[^.\n]{0,60}\b(to|at)\b[^.\n]{0,20}https?://|\bfetch_url\b[^.\n]{0,40}https?://", re.IGNORECASE),
    "tool_hijack": re.compile(r"\b(call|use|invoke|run|execute)\b[^.\n]{0,20}\b(the )?(fetch_url|calculator|search_documents|tool)\b", re.IGNORECASE),
}


def normalize_for_scan(text: str) -> str:
    """Defeat trivial obfuscation: zero-width characters, look-alike letters, odd spacing."""
    t = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    return re.sub(r"[ \t]+", " ", t)


def scan_injection(text: str) -> list[str]:
    """Names of the injection patterns found in text (empty when it looks clean)."""
    t = normalize_for_scan(text)
    return [name for name, pat in INJECTION_PATTERNS.items() if pat.search(t)]


_RESERVED = re.compile(r"<(/?)\s*(sources?|conversation|document|passage|untrusted_web_content)\b", re.IGNORECASE)


def neutralize_tags(text: str) -> str:
    """Document text must not be able to close or forge our own delimiters."""
    return _RESERVED.sub(lambda m: f"&lt;{m.group(1)}{m.group(2)}", text)


# --- output sanitizing -------------------------------------------------------------------------

_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_HTML_IMG = re.compile(r"<\s*img\b[^>]*>", re.IGNORECASE)
_MD_LINK = re.compile(r"\[([^\]]+)\]\(\s*([^)\s]+)[^)]*\)")
_REF_DEF = re.compile(r"^\s*\[[^\]]+\]:\s*\S+.*$", re.MULTILINE)
_BARE_URL = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)


def _domain(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def sanitize_output(text: str, allowed_domains: set[str] | frozenset[str] = frozenset()) -> str:
    """Remove markdown/HTML images and links to unknown domains from a model answer.

    An injected instruction typically exfiltrates data by making the answer render an image or a
    link whose URL carries the secret. Citations use [c_N] markers, so legitimate answers need no
    URLs at all; only domains of pages the user supplied or the agent was asked to fetch pass.
    """
    out = _MD_IMAGE.sub(lambda m: f"[image removed{': ' + m.group(1) if m.group(1) else ''}]", text)
    out = _HTML_IMG.sub("[image removed]", out)
    out = _REF_DEF.sub("", out)

    def link(m: re.Match[str]) -> str:
        label, url = m.group(1), m.group(2)
        if _domain(url) in allowed_domains:
            return m.group(0)
        return label  # keep the words, drop the destination

    out = _MD_LINK.sub(link, out)

    def bare(m: re.Match[str]) -> str:
        url = m.group(0)
        if _domain(url) in allowed_domains:
            return url
        # a URL with a query string is the usual carrier for leaked data
        return url.split("?")[0].split("#")[0] if "?" in url or "#" in url else url

    return _BARE_URL.sub(bare, out).strip()


# --- PII and secret masking --------------------------------------------------------------------

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]\d{3,4}[\s.-]\d{3,4}(?!\w)")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SECRET = re.compile(
    r"\b(?:sk-ant-[\w-]{8,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|xox[bp]-[\w-]{10,}"
    r"|eyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,})\b"
)
_BEARER = re.compile(r"(?i)\b(bearer|authorization:?|x-api-key:?|api[_-]?key[=:])\s*[\w.\-+/=]{8,}")


def _luhn(digits: str) -> bool:
    nums = [int(d) for d in digits[::-1]]
    total = sum(nums[0::2]) + sum(sum(divmod(d * 2, 10)) for d in nums[1::2])
    return total % 10 == 0


def redact_secrets(text: str) -> str:
    """Keys, tokens and credentials only: safe to apply to every log line."""
    out = _SECRET.sub("[REDACTED_KEY]", text)
    return _BEARER.sub(lambda m: f"{m.group(1)} [REDACTED_KEY]", out)


def mask_pii(text: str) -> str:
    """Emails, phone numbers, SSNs, valid card numbers and secrets. Used before anything that a
    person or a third-party tracing service could read (query logs, Langfuse)."""
    out = redact_secrets(text)
    out = _EMAIL.sub("[EMAIL]", out)
    out = _SSN.sub("[SSN]", out)

    def card(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return "[CARD]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)

    out = _CARD.sub(card, out)
    return _PHONE.sub("[PHONE]", out)


class _RedactingFactory:
    """Log record factory that redacts secrets, so a key can never reach a log by accident."""

    def __init__(self, inner):  # type: ignore[no-untyped-def]
        self._inner = inner

    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        record = self._inner(*args, **kwargs)
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 - never break logging
            return record
        redacted = redact_secrets(msg)
        if redacted != msg:
            record.msg, record.args = redacted, ()
        return record


_installed = False


def install_log_redaction() -> None:
    global _installed
    if not _installed:
        logging.setLogRecordFactory(_RedactingFactory(logging.getLogRecordFactory()))
        _installed = True
