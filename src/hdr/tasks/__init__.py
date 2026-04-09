"""
HDR Tasks - Standard library of common task types.
"""

from hdr.tasks.std import (
    ConceptDescribed,
    DirectoryCreated,
    FileWritten,
    MarkdownFileWritten,
)
from hdr.tasks.coding import PythonFileWritten, PythonWorkspaceBuilt
from hdr.tasks.meta import TaskCreated, FieldSpec, VerifySpec

__all__ = [
    "ConceptDescribed",
    "DirectoryCreated",
    "FileWritten",
    "MarkdownFileWritten",
    "PythonFileWritten",
    "PythonWorkspaceBuilt",
    "TaskCreated",
    "FieldSpec",
    "VerifySpec",
]
