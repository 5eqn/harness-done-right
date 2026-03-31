"""
Standard task types for common use cases.

These tasks cover typical file operations, content validation, and transformations.
All tasks use relative paths by preference, as they are more portable and make
projects easier to share and version control.
"""
from .. import BaseModel, verify, quote


class File(BaseModel):
    """
    Validates that a file exists at the given path.

    Prefer using relative paths for portability. The path can be absolute
    if needed, but relative paths are recommended for project-agnostic code.
    """
    path: str
    exists: bool = True

    def __init__(self, **data):
        super().__init__(**data)
        if self.exists:
            verify(f"File at {quote(self.path)} exists on the filesystem")
        else:
            verify(f"File at {quote(self.path)} does not exist on the filesystem")
