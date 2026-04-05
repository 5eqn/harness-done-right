"""
HDR - Harness Done Right

A structured task execution framework for Claude Code.
Formalize tasks as Python classes, validate with LLM assertions,
and execute incrementally.
"""

from pydantic import BaseModel

from hdr.tasks.coding import PythonFileWritten, PythonWorkspaceBuilt
from hdr.tasks.std import (
    ConceptDescribed,
    DirectoryCreated,
    FileWritten,
    MarkdownFileWritten,
    Task,
    quote,
)

__all__ = [
    "BaseModel",
    "Task",
    "quote",
    "FileWritten",
    "MarkdownFileWritten",
    "PythonFileWritten",
    "DirectoryCreated",
    "PythonWorkspaceBuilt",
    "ConceptDescribed",
]
