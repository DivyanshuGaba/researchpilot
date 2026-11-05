import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlparse

from app.services.ingestion.types import Block, ParsedDocument


class ParseError(Exception):
    pass


def sniff_type(head: bytes) -> str | None:
    """Detect file type from magic bytes, not the extension."""
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        return "docx"  # confirmed during parsing
    return None


def parse_pdf(path: Path) -> ParsedDocument:
    import fitz  # PyMuPDF

    blocks: list[Block] = []
    try:
        doc = fitz.open(path)
    except Exception as exc:  # corrupt or encrypted
        raise ParseError(f"cannot open PDF: {exc}") from exc
    with doc:
        for page_no, page in enumerate(doc, start=1):
            for x0, y0, x1, y1, text, _n, btype in page.get_text("blocks"):
                text = " ".join(str(text).split())
                if btype == 0 and text:
                    blocks.append(Block(text, page_no, (x0, y0, x1, y1)))
        title = (doc.metadata or {}).get("title") or path.stem
        pages = len(doc)
    if not blocks:
        raise ParseError("no extractable text (scanned PDF? OCR is not enabled yet)")
    return ParsedDocument(title=title, blocks=blocks, page_count=pages)


def parse_docx(path: Path) -> ParsedDocument:
    import docx

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise ParseError(f"cannot open DOCX: {exc}") from exc
    blocks: list[Block] = []
    heading_path: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name if para.style is not None else ""
        if style.startswith("Heading"):
            level = int(style.split()[-1]) if style.split()[-1].isdigit() else 1
            heading_path = heading_path[: level - 1] + [text]
            blocks.append(Block(text, 1, None, "heading", tuple(heading_path)))
        else:
            blocks.append(Block(text, 1, None, "paragraph", tuple(heading_path)))
    if not blocks:
        raise ParseError("DOCX has no text")
    # DOCX has no fixed pagination; everything is reported as page 1.
    return ParsedDocument(title=path.stem, blocks=blocks, page_count=1)


def assert_public_url(url: str) -> None:
    """SSRF guard: https/http only, and every resolved address must be public."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ParseError("only http(s) URLs are allowed")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ParseError("cannot resolve host") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise ParseError("URL resolves to a non-public address")


def parse_url(url: str) -> ParsedDocument:
    import trafilatura

    assert_public_url(url)
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded) if downloaded else None
    if not text:
        raise ParseError("could not extract content from URL")
    blocks = [Block(p.strip(), 1) for p in text.split("\n") if p.strip()]
    return ParsedDocument(title=url, blocks=blocks, page_count=1)


def parse(source_type: str, location: str) -> ParsedDocument:
    if source_type == "pdf":
        return parse_pdf(Path(location))
    if source_type == "docx":
        return parse_docx(Path(location))
    if source_type == "url":
        return parse_url(location)
    raise ParseError(f"unsupported source type {source_type}")
