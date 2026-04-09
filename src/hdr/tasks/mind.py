"""
Humanities and mind-related task types for HDR.

These tasks cover conceptual documentation, knowledge management, and human-centric
validation use cases that require qualitative judgment.
"""

from pydantic import Field
from typing import TYPE_CHECKING

from hdr.tasks.std import Task

if TYPE_CHECKING:
    from hdr.tasks.coding import MarkdownFileWritten


class ConceptDescribed(Task):
    """
    Represents a documented concept within a context.
    """

    context: "MarkdownFileWritten" = Field(
        description="File explaining the parent context"
    )
    name: str = Field(description="Name of the concept")
    description: "MarkdownFileWritten" = Field(
        description="File containing the concept description"
    )

    def __init__(self, **data):
        super().__init__(**data)

        self.verify(
            "The description is written for readers who understand context but do not yet know name; it neither repeats basics from context nor presumes knowledge of sibling/descendant concepts."
        )
        self.verify(
            "The concept name represents exactly one atomic idea that cannot be meaningfully split into two independent concepts."
        )
        self.verify(
            "The description contains no time-sensitive terms (e.g., 'currently', 'recently', 'as of now') without specifying an exact version or date."
        )
        self.verify(
            "The description identifies (a) a broader category that name belongs to, and (b) a distinguishing property that separates it from other members of that category."
        )
        self.verify(
            "A reader familiar with context can determine for any concrete instance whether it belongs to name, with at most minor edge-case ambiguity."
        )
