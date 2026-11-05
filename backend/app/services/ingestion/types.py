from dataclasses import dataclass, field


def count_tokens(text: str) -> int:
    """Cheap approximation (about 4 chars per token). Avoids a tokenizer download at runtime."""
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class Block:
    text: str
    page: int
    bbox: tuple[float, float, float, float] | None = None
    block_type: str = "paragraph"  # heading, paragraph, table, list, code
    heading_path: tuple[str, ...] = ()


@dataclass
class ParsedDocument:
    title: str
    blocks: list[Block]
    page_count: int


@dataclass
class ChunkDraft:
    level: str  # parent | child
    index: int
    text: str
    page_start: int
    page_end: int
    bbox_list: list[dict[str, object]] = field(default_factory=list)
    heading_path: list[str] = field(default_factory=list)
    parent_index: int | None = None

    @property
    def token_count(self) -> int:
        return count_tokens(self.text)
