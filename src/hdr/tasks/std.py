"""
Standard task types for common use cases.

These tasks cover typical file operations, content validation, and transformations.
All tasks use relative paths by preference, as they are more portable and make
projects easier to share and version control.
"""

from hdr import BaseModel, verify, quote

import os
import shutil
import subprocess


class File(BaseModel):
    """
    Validates that a file exists at the given path using os.path.exists().

    Prefer using relative paths for portability. The path can be absolute
    if needed, but relative paths are recommended for project-agnostic code.

    The `content` field is auto-filled from the actual file content if not specified.
    """

    path: str
    content: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        if not os.path.exists(self.path):
            raise AssertionError(f"File at {self.path} does not exist")
        # Auto-fill content from actual file if not provided
        if not self.content:
            try:
                with open(self.path, "r") as f:
                    self.content = f.read()
            except (IOError, OSError):
                raise AssertionError(f"Could not read file at {self.path}")


class Directory(BaseModel):
    """
    Validates that a directory exists at the given path using os.path.isdir().

    The `content` field is auto-filled from the actual directory content if not specified.
    Content is a list of File objects representing the files in the directory,
    gathered recursively and respecting .gitignore patterns.
    """

    path: str
    content: list[File] = []

    def __init__(self, **data):
        super().__init__(**data)
        if not os.path.isdir(self.path):
            raise AssertionError(f"Directory at {self.path} does not exist")
        # Auto-fill content from actual directory if not provided
        if not self.content:
            self.content = self._gather_content(self.path)
            total_files = len(self.content)
            print(f"[Directory] Total files in {self.path}: {total_files}")

    def _gather_content(self, dir_path: str) -> list[File]:
        """Gather content from directory as list[File], respecting .gitignore and recursing."""
        files: list[File] = []
        gitignore_path = os.path.join(dir_path, ".gitignore")
        gitignore_patterns = []
        if os.path.exists(gitignore_path):
            try:
                with open(gitignore_path, "r") as f:
                    gitignore_patterns = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    ]
            except (IOError, OSError):
                pass

        for root, dirs, filenames in os.walk(dir_path):
            # Calculate relative path for filtering directories
            rel_root = os.path.relpath(root, dir_path)
            # Filter out directories matching gitignore patterns
            dirs[:] = [
                d
                for d in dirs
                if not self._is_ignored(os.path.join(rel_root, d), gitignore_patterns)
            ]

            for filename in filenames:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, dir_path)
                # Check if file should be ignored
                if self._is_ignored(rel_path, gitignore_patterns):
                    continue
                try:
                    with open(filepath, "r") as f:
                        file_content = f.read()
                    files.append(File(path=filepath, content=file_content))
                except (IOError, OSError):
                    pass

        return files

    def _is_ignored(self, rel_path: str, patterns: list[str]) -> bool:
        """Check if a path matches any gitignore pattern."""
        import fnmatch

        for pattern in patterns:
            if pattern.endswith("/"):
                # Directory pattern
                dir_pattern = pattern.rstrip("/")
                if fnmatch.fnmatch(rel_path, dir_pattern) or fnmatch.fnmatch(
                    rel_path, dir_pattern + "/*"
                ):
                    return True
            else:
                if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(
                    os.path.basename(rel_path), pattern
                ):
                    return True
        return False


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
            raise AssertionError(
                f"pyright produced unexpected output in {self.path} "
                f"(exit code {result_pyright.returncode}). Could not verify. Run 'pyright' for details."
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


class Concept(BaseModel):
    context: str
    name: str
    description: File

    def __init__(self, **data):
        super().__init__(**data)

        ctx = f"[Concept Definition] context={quote(self.context)} name={quote(self.name)} description={quote(self.description)} [Verify]"

        verify(
            f"{ctx} The description is written for readers who understand context but do not yet know name; it neither repeats basics from context nor presumes knowledge of sibling/descendant concepts."
        )
        verify(
            f"{ctx} The concept name represents exactly one atomic idea that cannot be meaningfully split into two independent concepts."
        )
        verify(
            f"{ctx} The description contains no time-sensitive terms (e.g., 'currently', 'recently', 'as of now') without specifying an exact version or date."
        )
        verify(
            f"{ctx} The description identifies (a) a broader category that name belongs to, and (b) a distinguishing property that separates it from other members of that category."
        )
        verify(
            f"{ctx} A reader familiar with context can determine for any concrete instance whether it belongs to name, with at most minor edge-case ambiguity."
        )


class Context(BaseModel):
    concepts: list[Concept]
