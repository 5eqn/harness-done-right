"""
Standard task types for common use cases.

These tasks cover typical file operations, content validation, and transformations.
All tasks use relative paths by preference, as they are more portable and make
projects easier to share and version control.
"""

import os

from hdr.core import BaseModel


class File(BaseModel):
    """
    Validates that a file exists at the given path using os.path.exists().

    Prefer using relative paths for portability. The path can be absolute
    if needed, but relative paths are recommended for project-agnostic code.

    Quoting this object (via quote()) will return the file's full content.
    """

    path: str
    exists: bool = True

    def __init__(self, **data):
        super().__init__(**data)
        file_exists = os.path.exists(self.path)
        if self.exists and not file_exists:
            raise AssertionError(f"File at {self.path} does not exist")
        if not self.exists and file_exists:
            raise AssertionError(f"File at {self.path} should not exist")

    def model_dump_json(self, **kwargs):  # noqa: ARG002
        content = ""
        if self.exists:
            try:
                with open(self.path, "r") as f:
                    content = f.read()
            except (IOError, OSError):
                pass
        return f"<file><path>{self.path}</path><content>{content}</content></file>"
