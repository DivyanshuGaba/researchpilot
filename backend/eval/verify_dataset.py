"""Check that every evidence phrase appears verbatim on its stated pages of the pinned PDFs, and
that the unanswerable questions' key terms are absent. Run after fetch_docs; fails loudly."""

import hashlib
import json
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).parent
ABSENT_TERMS: dict[str, dict[str, list[str]]] = {
    # question id -> {document key or "*" for all documents: terms that must not appear there}
    "una-01": {"*": ["Mongolia", "Ulaanbaatar"]}, "una-02": {"*": ["Pride and Prejudice", "Austen"]},
    "una-03": {"*": ["sourdough"]}, "una-04": {"*": ["castling"]}, "una-05": {"*": ["tungsten"]},
    "una-06": {"*": ["CO2", "kilowatt", "carbon footprint"]}, "una-07": {"*": ["billion parameters"]},
    "una-08": {"rag": ["learning rate", "learning-rate"]},  # other papers do state theirs
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def load_pages() -> dict[str, list[str]]:
    manifest = json.loads((ROOT / "docs.json").read_text())
    out = {}
    for d in manifest:
        path = ROOT / "docs" / f"{d['key']}.pdf"
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != d["sha256"]:
            sys.exit(f"{d['key']}: sha256 mismatch, the PDF is not the pinned version")
        with pymupdf.open(path) as doc:
            out[d["key"]] = [norm(doc.load_page(i).get_text()) for i in range(doc.page_count)]
    return out


def main() -> int:
    pages = load_pages()
    rows = [json.loads(line) for line in (ROOT / "dataset.jsonl").read_text().splitlines() if line.strip()]
    problems = []
    for r in rows:
        for s in r["sources"]:
            for p in s["pages"]:
                if norm(s["evidence"]) not in pages[s["doc"]][p - 1]:
                    problems.append(f"{r['id']}: evidence not on {s['doc']} p{p}: {s['evidence'][:70]!r}")
        for scope, terms in ABSENT_TERMS.get(r["id"], {}).items():
            for key, plist in pages.items():
                if scope not in ("*", key):
                    continue
                for term in terms:
                    for i, text in enumerate(plist, 1):
                        if term.lower() in text.lower():
                            problems.append(f"{r['id']}: {term!r} unexpectedly present in {key} p{i}")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids")
    n_un = sum(r["type"] == "unanswerable" for r in rows)
    if len(rows) < 30 or n_un < 5:
        problems.append(f"need >= 30 rows and >= 5 unanswerable, have {len(rows)} and {n_un}")
    for p in problems:
        print("PROBLEM:", p)
    print(f"{len(rows)} questions, {n_un} unanswerable, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
