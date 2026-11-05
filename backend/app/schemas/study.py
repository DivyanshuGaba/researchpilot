import re
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

_SOURCE = re.compile(r"^s_\d+$")


class QuizQuestionDraft(BaseModel):
    """What the model must return for one question (validated before anything is stored)."""

    question: str = Field(min_length=5, max_length=600)
    options: list[str] = Field(min_length=4, max_length=4)
    answer_index: int = Field(ge=0, le=3)
    explanation: str = Field(min_length=3, max_length=800)
    source: str

    @field_validator("options")
    @classmethod
    def _options_ok(cls, v: list[str]) -> list[str]:
        cleaned = [o.strip() for o in v]
        if any(not o or len(o) > 400 for o in cleaned):
            raise ValueError("options must be non-empty and under 400 characters")
        if len({o.lower() for o in cleaned}) != len(cleaned):
            raise ValueError("options must be distinct")
        return cleaned

    @field_validator("source")
    @classmethod
    def _source_ok(cls, v: str) -> str:
        if not _SOURCE.match(v):
            raise ValueError("source must look like s_1")
        return v


class QuizDraft(BaseModel):
    questions: list[QuizQuestionDraft] = Field(min_length=1)


class CardDraft(BaseModel):
    front: str = Field(min_length=2, max_length=400)
    back: str = Field(min_length=1, max_length=600)
    source: str

    @model_validator(mode="after")
    def _source_ok(self) -> "CardDraft":
        if not _SOURCE.match(self.source):
            raise ValueError("source must look like s_1")
        return self


class CardsDraft(BaseModel):
    cards: list[CardDraft] = Field(min_length=1)


# --- API models -------------------------------------------------------------------------------


class QuizRequest(BaseModel):
    document_id: uuid.UUID
    num_questions: int = Field(default=5, ge=1, le=20)


class FlashcardRequest(BaseModel):
    document_id: uuid.UUID
    num_cards: int = Field(default=10, ge=1, le=30)


class SummaryRequest(BaseModel):
    document_id: uuid.UUID


class QuizQuestionOut(BaseModel):
    question: str
    options: list[str]
    answer_index: int
    explanation: str
    chunk_id: uuid.UUID
    page_start: int
    page_end: int


class QuizOut(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    requested: int
    questions: list[QuizQuestionOut]
    created_at: datetime


class CardOut(BaseModel):
    front: str
    back: str
    chunk_id: uuid.UUID
    page_start: int
    page_end: int


class FlashcardSetOut(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    requested: int
    cards: list[CardOut]
    created_at: datetime


class SummaryOut(BaseModel):
    document_id: uuid.UUID
    summary: str
    cached: bool
    sections: int
    truncated: bool = False
    meta: dict[str, Any] = {}
