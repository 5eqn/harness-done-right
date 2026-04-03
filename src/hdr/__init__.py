"""
HDR - Harness Done Right

A structured task execution framework for Claude Code.
Formalize tasks as Python classes, validate with LLM assertions, and execute incrementally.
"""

from pydantic import BaseModel

from hdr.verifiers.std import verify, quote
from hdr.tasks.std import Directory, File, PythonWorkspace, Context, Concept

__all__ = [
    "BaseModel",
    "verify",
    "quote",
    "File",
    "Directory",
    "PythonWorkspace",
    "Context",
    "Concept",
]
