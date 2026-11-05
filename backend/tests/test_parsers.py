import pytest

from app.services.ingestion import parsers


def test_sniff_type_uses_magic_bytes():
    assert parsers.sniff_type(b"%PDF-1.7") == "pdf"
    assert parsers.sniff_type(b"PK\x03\x04rest") == "docx"
    assert parsers.sniff_type(b"MZ\x90\x00") is None


def test_parse_pdf_keeps_page_numbers(tmp_path):
    import fitz

    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    for text in ("First page about cats.", "Second page about dogs."):
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    parsed = parsers.parse_pdf(path)
    assert parsed.page_count == 2
    pages = {b.page: b.text for b in parsed.blocks}
    assert "cats" in pages[1] and "dogs" in pages[2]
    assert all(b.bbox is not None for b in parsed.blocks)


def test_parse_pdf_without_text_fails_clearly(tmp_path):
    import fitz

    path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(path)
    with pytest.raises(parsers.ParseError):
        parsers.parse_pdf(path)


def test_parse_docx_tracks_headings(tmp_path):
    import docx

    path = tmp_path / "doc.docx"
    d = docx.Document()
    d.add_heading("Methods", level=1)
    d.add_paragraph("We did things.")
    d.save(str(path))
    parsed = parsers.parse_docx(path)
    body = next(b for b in parsed.blocks if b.block_type == "paragraph")
    assert body.heading_path == ("Methods",)


@pytest.mark.parametrize(
    "url", ["ftp://example.com/x", "http://127.0.0.1/admin", "http://localhost:8000", "http://169.254.169.254/"]
)
def test_ssrf_guard_blocks_non_public(url):
    with pytest.raises(parsers.ParseError):
        parsers.assert_public_url(url)


def test_control_characters_from_math_glyphs_are_removed():
    from app.services.ingestion.parsers import clean_text

    assert clean_text("p(z|x) = exp\x00 d(z)\x01 q(x)\x1f") == "p(z|x) = exp d(z) q(x)"
    assert clean_text("keeps\ttabs\nand newlines") == "keeps\ttabs\nand newlines"
    assert clean_text("é ∑ ⊤ 日本語") == "é ∑ ⊤ 日本語"  # real unicode is untouched
