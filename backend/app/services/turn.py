from dataclasses import dataclass, field

from app.services.vectorstore import Hit


@dataclass
class Group:
    """Hits for one document in compare mode, or the single anonymous group in QA mode."""

    title: str | None
    hits: list[Hit]


@dataclass
class Turn:
    """Everything generation needs, produced by retrieval, rewrite and memory."""

    question: str
    mode: str = "qa"  # qa | compare
    standalone: str = ""
    history_text: str = ""
    groups: list[Group] = field(default_factory=list)
    gated: bool = False
    closest: list[Hit] = field(default_factory=list)
    top_score: float | None = None
    timings_ms: dict[str, int] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)
