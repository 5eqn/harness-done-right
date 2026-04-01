"""
HDR - Harness Done Right

A structured task execution framework for Claude Code.
Formalize tasks as Python classes, validate with LLM assertions, and execute incrementally.
"""

from pydantic import BaseModel

from hdr.core import (
    verify,
    quote,
    set_mock_mode,
)

__all__ = [
    "BaseModel",
    "verify",
    "quote",
    "set_mock_mode",
]
