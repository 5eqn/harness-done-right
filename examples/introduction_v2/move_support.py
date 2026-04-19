from enum import Enum

from pydantic import BaseModel, Field

from hdr.contracts.coding import MarkdownFile


class MovePurpose(str, Enum):
    BUILD = "build"
    SHIFT = "shift"


class MoveSnapshot(BaseModel):
    purpose: MovePurpose = Field(description="Purpose of a previously chosen move")
    content: str = Field(description="Atomic outline sentence chosen for that move")
    contrast: str = Field(description="Single projected contrast axis for that move")
    before: str = Field(description="Relevant reader lens before that move")
    after: str = Field(description="Relevant reader lens after that move")


class MoveContext(BaseModel):
    target_reader: MarkdownFile = Field(
        description="Markdown file describing what the intended reader already knows"
    )
    target_concept: str = Field(description="Concept the move chain is introducing")
    target_meaning: MarkdownFile = Field(
        description="Markdown file describing the full conceptual payload the chain must express"
    )
    prior_moves: list[MoveSnapshot] = Field(
        default_factory=list,
        description="Previously chosen moves, stored as snapshots to avoid recursive generation",
    )
