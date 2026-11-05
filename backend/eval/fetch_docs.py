"""Download the pinned evaluation papers (public arXiv PDFs) and verify their SHA-256.

    python -m eval.fetch_docs
"""

import hashlib
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).parent


def main() -> int:
    manifest = json.loads((ROOT / "docs.json").read_text())
    (ROOT / "docs").mkdir(exist_ok=True)
    bad = 0
    for d in manifest:
        path = ROOT / "docs" / f"{d['key']}.pdf"
        if not path.exists():
            r = httpx.get(d["pdf"], follow_redirects=True, timeout=90)
            r.raise_for_status()
            path.write_bytes(r.content)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ok = digest == d["sha256"]
        bad += not ok
        print(f"{d['key']:20} {'ok' if ok else 'HASH MISMATCH'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
