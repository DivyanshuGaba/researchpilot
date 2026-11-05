from app.config import PipelineConfig
from app.services.ingestion.chunkers import RecursiveChunker
from app.services.ingestion.types import Block, count_tokens


def _blocks(n: int, page_every: int = 5) -> list[Block]:
    return [
        Block(f"Sentence number {i} talks about topic {i}. " * 6, page=1 + i // page_every,
              bbox=(0, 0, 10, 10))
        for i in range(n)
    ]


def test_children_reference_valid_parents_and_respect_size():
    cfg = PipelineConfig(child_tokens=100, parent_tokens=400, overlap_tokens=20)
    drafts = RecursiveChunker(cfg).chunk(_blocks(30))
    parents = [d for d in drafts if d.level == "parent"]
    children = [d for d in drafts if d.level == "child"]
    assert parents and children
    for c in children:
        assert c.parent_index is not None and 0 <= c.parent_index < len(parents)
        assert c.token_count <= cfg.child_tokens + cfg.overlap_tokens + 80  # one atom of slack
    for p in parents:
        assert p.token_count <= cfg.parent_tokens + cfg.child_tokens


def test_page_range_is_tracked():
    drafts = RecursiveChunker(PipelineConfig()).chunk(_blocks(20, page_every=5))
    parent = next(d for d in drafts if d.level == "parent")
    assert parent.page_start == 1
    assert parent.page_end >= parent.page_start
    assert all(d.page_start >= 1 for d in drafts)


def test_children_never_span_pages():
    blocks = [Block(f"short block {i}", page=i + 1) for i in range(4)]
    children = [d for d in RecursiveChunker(PipelineConfig()).chunk(blocks) if d.level == "child"]
    assert [(c.page_start, c.page_end) for c in children] == [(1, 1), (2, 2), (3, 3), (4, 4)]


def test_tables_are_never_split():
    table = Block("| a | b |\n" * 400, page=2, block_type="table")
    cfg = PipelineConfig(child_tokens=50, parent_tokens=100)
    drafts = RecursiveChunker(cfg).chunk([table])
    children = [d for d in drafts if d.level == "child"]
    assert len(children) == 1
    assert children[0].text == table.text


def test_overlap_repeats_tail_atoms():
    cfg = PipelineConfig(child_tokens=60, parent_tokens=10_000, overlap_tokens=30)
    blocks = [Block(f"unique-{i} " + "x" * 80, page=1) for i in range(12)]
    children = [d for d in RecursiveChunker(cfg).chunk(blocks) if d.level == "child"]
    assert len(children) > 1
    assert children[1].text.split("\n")[0] in children[0].text


def test_long_single_block_is_split():
    big = Block("word " * 3000, page=1)
    drafts = RecursiveChunker(PipelineConfig(child_tokens=100)).chunk([big])
    assert all(count_tokens(d.text) <= 400 for d in drafts if d.level == "child")
    assert len([d for d in drafts if d.level == "child"]) > 5


class TopicEmbedder:
    """Maps cat sentences and rocket sentences to orthogonal vectors."""

    model_id, dim = "topic", 2

    def embed_documents(self, texts):
        return [[1.0, 0.0] if "cat" in t else [0.0, 1.0] for t in texts]

    def embed_query(self, text):  # pragma: no cover
        return [1.0, 0.0]


def test_semantic_chunker_splits_at_topic_shift():
    from app.services.ingestion.chunkers import SemanticChunker

    cats = " ".join(f"The cat number {i} sits on the warm mat today." for i in range(3))
    rockets = " ".join(f"Rocket number {i} launches into orbit at dawn." for i in range(3))
    cfg = PipelineConfig(chunker="semantic", child_tokens=100, parent_tokens=1000)
    drafts = SemanticChunker(cfg, TopicEmbedder()).chunk([Block(cats + " " + rockets, page=1)])
    children = [d for d in drafts if d.level == "child"]
    assert len(children) == 2
    assert "cat" in children[0].text and "Rocket" not in children[0].text
    assert "Rocket" in children[1].text and "cat" not in children[1].text
    assert all(c.parent_index == 0 for c in children)


def test_semantic_chunker_respects_pages_and_size():
    from app.services.ingestion.chunkers import SemanticChunker

    blocks = [Block(f"The cat number {i} sits on the mat. " * 3, page=1 + i // 2) for i in range(4)]
    cfg = PipelineConfig(chunker="semantic", child_tokens=60, parent_tokens=1000)
    children = [d for d in SemanticChunker(cfg, TopicEmbedder()).chunk(blocks) if d.level == "child"]
    assert all(c.page_start == c.page_end for c in children)
    assert all(c.token_count <= 60 + 20 for c in children)


def test_get_chunker_selects_by_config_and_requires_embedder():
    import pytest

    from app.services.ingestion.chunkers import SemanticChunker, get_chunker

    assert isinstance(get_chunker(PipelineConfig(chunker="semantic"), TopicEmbedder()), SemanticChunker)
    assert isinstance(get_chunker(PipelineConfig()), RecursiveChunker)
    with pytest.raises(ValueError):
        get_chunker(PipelineConfig(chunker="semantic"))
