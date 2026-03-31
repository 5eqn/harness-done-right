"""
Standard task types for common use cases.

These tasks cover typical file operations, content validation, and transformations.
All tasks use relative paths by preference, as they are more portable and make
projects easier to share and version control.
"""
from hdr import BaseModel, verify, quote


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


class FileContent(BaseModel):
    """
    Validates that a file contains expected content.

    Useful for verifying that code, documentation, or configuration files
    have been properly created with the right content.
    """
    path: str
    expected_content: str

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"File at {quote(self.path)} contains the expected content")
        verify(f"The content in {quote(self.path)} matches exactly what was specified")


class Readme(BaseModel):
    """
    Validates that a README.md file exists and is properly formatted.

    A good README should explain what the project does, how to install it,
    and how to use it.
    """
    path: str = "README.md"

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"README file at {quote(self.path)} exists")
        verify(f"README at {quote(self.path)} contains a clear project description")
        verify(f"README at {quote(self.path)} contains installation instructions")
        verify(f"README at {quote(self.path)} contains usage instructions")


class PythonFile(BaseModel):
    """
    Validates that a Python file exists and is syntactically valid.

    Does not execute the code, only checks that it can be parsed as valid Python.
    """
    path: str

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"Python file at {quote(self.path)} exists and is valid Python syntax")


class TestFile(BaseModel):
    """
    Validates that a test file exists and is properly structured.

    A good test file should have descriptive test functions and use
    standard Python testing frameworks like pytest.
    """
    path: str
    test_functions: list[str]

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"Test file at {quote(self.path)} exists")
        verify(f"Test file at {quote(self.path)} contains the expected test functions")


class Documentation(BaseModel):
    """
    Validates that documentation exists and covers essential topics.

    Good documentation should explain what something is, why you'd use it,
    and how to get started.
    """
    path: str
    title: str

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"Documentation at {quote(self.path)} exists")
        verify(f"Documentation at {quote(self.path)} has the title {quote(self.title)}")
        verify(f"Documentation at {quote(self.path)} is clear and well-structured")
