import re
from typing import Protocol

from app.config import PipelineConfig
from app.services.ingestion.types import Block, ChunkDraft, count_tokens

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_UNSPLITTABLE = {"table", "code"}


class Chunker(Protocol):
    def chunk(self, blocks: list[Block]) -> list[ChunkDraft]: ...


def _split_text(text: str, max_tokens: int) -> list[str]:
    """Split on sentences, falling back to words for a sentence that is still too long."""
    pieces: list[str] = []
    for sentence in _SENTENCE.split(text):
        if count_tokens(sentence) <= max_tokens:
            pieces.append(sentence)
            continue
        words, current = sentence.split(), []
        for word in words:
            current.append(word)
            if count_tokens(" ".join(current)) >= max_tokens:
                pieces.append(" ".join(current))
                current = []
        if current:
            pieces.append(" ".join(current))
    return pieces


def _atoms(blocks: list[Block], max_tokens: int) -> list[Block]:
    out: list[Block] = []
    for block in blocks:
        if block.block_type in _UNSPLITTABLE or count_tokens(block.text) <= max_tokens:
            out.append(block)
            continue
        for piece in _split_text(block.text, max_tokens):
            out.append(Block(piece, block.page, block.bbox, block.block_type, block.heading_path))
    return out


def _group(atoms: list[Block], limit: int, overlap: int = 0) -> list[list[Block]]:
    groups: list[list[Block]] = []
    current: list[Block] = []
    size = 0
    for atom in atoms:
        t = count_tokens(atom.text)
        if current and size + t > limit:
            groups.append(current)
            carried: list[Block] = []
            carried_size = 0
            for prev in reversed(current):  # carry a tail of atoms as overlap
                pt = count_tokens(prev.text)
                if carried_size + pt > overlap:
                    break
                carried.insert(0, prev)
                carried_size += pt
            current, size = carried, carried_size
        current.append(atom)
        size += t
    if current:
        groups.append(current)
    return groups


def _draft(level: str, index: int, group: list[Block], parent: int | None = None) -> ChunkDraft:
    bboxes: list[dict[str, object]] = [
        {"page": b.page, "bbox": list(b.bbox)} for b in group if b.bbox is not None
    ]
    heading = list(group[0].heading_path)
    return ChunkDraft(
        level=level,
        index=index,
        text="\n".join(b.text for b in group),
        page_start=min(b.page for b in group),
        page_end=max(b.page for b in group),
        bbox_list=bboxes,
        heading_path=heading,
        parent_index=parent,
    )


class RecursiveChunker:
    """Parent/child small-to-big chunking. Children are embedded; parents go to the LLM."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def chunk(self, blocks: list[Block]) -> list[ChunkDraft]:
        c = self.config
        atoms = _atoms(blocks, c.child_tokens)
        drafts: list[ChunkDraft] = []
        parents = _group(atoms, c.parent_tokens)
        parent_drafts = [_draft("parent", i, g) for i, g in enumerate(parents)]
        drafts.extend(parent_drafts)
        child_index = 0
        for p_idx, parent_atoms in enumerate(parents):
            for group in _group(parent_atoms, c.child_tokens, c.overlap_tokens):
                drafts.append(_draft("child", child_index, group, parent=p_idx))
                child_index += 1
        return drafts


def get_chunker(config: PipelineConfig) -> Chunker:
    if config.chunker == "recursive":
        return RecursiveChunker(config)
    raise ValueError(f"unknown chunker {config.chunker!r}")
