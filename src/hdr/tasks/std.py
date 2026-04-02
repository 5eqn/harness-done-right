"""
Standard task types for common use cases.

These tasks cover typical file operations, content validation, and transformations.
All tasks use relative paths by preference, as they are more portable and make
projects easier to share and version control.
"""

import os
import shutil
import subprocess

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


class Directory(BaseModel):
    """
    Validates that a directory exists at the given path using os.path.isdir().

    The optional `files` parameter lists files (relative to the directory) that
    must also exist. Each file path is joined with `path` before checking.

    Quoting this object will return the directory path and its files list.
    """

    path: str
    exists: bool = True
    files: list[str] = []

    def __init__(self, **data):
        super().__init__(**data)
        dir_exists = os.path.isdir(self.path)
        if self.exists and not dir_exists:
            raise AssertionError(f"Directory at {self.path} does not exist")
        if not self.exists and dir_exists:
            raise AssertionError(f"Directory at {self.path} should not exist")
        for rel_file in self.files:
            file_path = os.path.join(self.path, rel_file)
            if not os.path.exists(file_path):
                raise AssertionError(f"File {file_path} does not exist")

    def model_dump_json(self, **kwargs):  # noqa: ARG002
        return f"<directory><path>{self.path}</path><files>{self.files}</files></directory>"


class PythonWorkspace(Directory):
    """
    Validates a Python workspace is properly configured for linting and type checking.

    Inherits from Directory — the directory must exist.
    Additionally verifies:
    - ruff is installed (shutil.which)
    - pyright is installed (shutil.which)
    - pyright reports no warnings or errors in the workspace
    - ruff check reports no lint errors in the workspace

    If ruff or pyright is not installed, raises AssertionError with
    an installation message instructing the caller.
    """

    def __init__(self, **data):
        super().__init__(**data)

        # Check ruff is installed
        ruff_path = shutil.which("ruff")
        if not ruff_path:
            raise AssertionError(
                "ruff is not installed. Please install it with: pip install ruff"
            )

        # Check pyright is installed
        pyright_path = shutil.which("pyright")
        if not pyright_path:
            raise AssertionError(
                "pyright is not installed. Please install it with: pip install pyright"
            )

        # Run pyright and check for no errors/warnings
        result_pyright = subprocess.run(
            ["pyright", "--outputjson"],
            cwd=self.path,
            capture_output=True,
            text=True,
        )
        try:
            import json
            pyright_output = json.loads(result_pyright.stdout)
            error_count = pyright_output.get("summary", {}).get("errorCount", 0)
            warning_count = pyright_output.get("summary", {}).get("warningCount", 0)
            if error_count > 0 or warning_count > 0:
                raise AssertionError(
                    f"pyright found {error_count} error(s) and {warning_count} warning(s) "
                    f"in {self.path}. Run 'pyright' for details."
                )
        except (json.JSONDecodeError, KeyError):
            if result_pyright.returncode != 0:
                raise AssertionError(
                    f"pyright failed in {self.path} (exit code {result_pyright.returncode}). "
                    f"Run 'pyright' for details."
                )

        # Run ruff check and verify no lint errors
        result_ruff = subprocess.run(
            ["ruff", "check", "."],
            cwd=self.path,
            capture_output=True,
            text=True,
        )
        if result_ruff.returncode != 0:
            raise AssertionError(
                f"ruff found lint errors in {self.path}:\n{result_ruff.stdout}\n{result_ruff.stderr}"
            )