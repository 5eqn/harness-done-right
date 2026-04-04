"""
HDR - Harness Done Right

A structured task execution framework for Claude Code.
Formalize tasks as Python classes, validate with LLM assertions,
and execute incrementally.
"""

from pydantic import BaseModel

from hdr.tasks.std import (
    Task,
    Directory,
    File,
    PythonWorkspace,
    Concept,
    quote,
)

__all__ = [
    "BaseModel",
    "Task",
    "quote",
    "File",
    "Directory",
    "PythonWorkspace",
    "Concept",
]
