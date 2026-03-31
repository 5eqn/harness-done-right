"""
HDR - Harness Done Right

A structured task execution framework for Claude Code.
Formalize tasks as Python classes, validate with LLM assertions, and execute incrementally.
"""
from pydantic import BaseModel

from hdr.core import (
    verify,
    quote,
    checkout,
    set_mock_mode,
    get_checkout_dir,
)

__all__ = [
    "BaseModel",
    "verify",
    "quote",
    "checkout",
    "set_mock_mode",
    "get_checkout_dir",
]
