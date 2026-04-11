"""
Coding tasks for Python-related validation.

These tasks validate Python files and workspaces, using tools like ruff and pyright.
"""

import shutil
import subprocess
from typing import Any
import os

from pydantic import model_validator

from hdr.tasks.std import DirectoryCreated, FileWritten


class PythonFileWritten(FileWritten):
    """
    Validates that a Python file exists and passes linting and type checking.
    Inherits all fields from FileWritten.

    Additionally verifies:
    - Path ends with `.py`
    - pyright reports no warnings or errors for the file
    - ruff check reports no lint errors for the file
    - ruff format reports clean formatting for the file
    """

    @model_validator(mode="before")
    @classmethod
    def _format_python_and_fill_content(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        data = dict(data)
        path = data.get("path")
        if not isinstance(path, str) or not path.endswith(".py"):
            return data

        cls._validate_python_file(path)
        data["_hdr_content"] = cls._read_file_content(path)
        return data

    def __init__(self, **data):
        super().__init__(**data)
        if not self.path.endswith(".py"):
            raise AssertionError(f"Path '{self.path}' does not end with '.py'")

    @staticmethod
    def _validate_python_file(path: str) -> None:
        if not os.path.exists(path):
            raise AssertionError(f"File at {path} does not exist")

        # Check pyright is installed
        pyright_path = shutil.which("pyright")
        if not pyright_path:
            raise AssertionError(
                "pyright is not installed. Please install it with: pip install pyright"
            )

        # Run pyright on the single file
        result_pyright = subprocess.run(
            ["pyright", "--outputjson", path],
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
                    f"in {path}. Run 'pyright {path}' for details."
                )
        except (json.JSONDecodeError, KeyError):
            raise AssertionError(
                f"pyright produced unexpected output for {path} "
                f"(exit code {result_pyright.returncode}). Could not verify. Run 'pyright {path}' for details."
            )

        # Run ruff check on the single file
        result_ruff = subprocess.run(
            ["ruff", "check", path],
            capture_output=True,
            text=True,
        )
        if result_ruff.returncode != 0:
            raise AssertionError(
                f"ruff found lint errors in {path}:\n{result_ruff.stdout}\n{result_ruff.stderr}"
            )

        # Run ruff format on the single file to ensure consistent formatting
        result_ruff_fmt = subprocess.run(
            ["ruff", "format", path],
            capture_output=True,
            text=True,
        )
        if result_ruff_fmt.returncode != 0:
            raise AssertionError(
                f"ruff format failed for {path}:\n{result_ruff_fmt.stdout}\n{result_ruff_fmt.stderr}"
            )


class MarkdownFileWritten(FileWritten):
    """
    Validates that a markdown file exists at the given path, auto-formats it, and checks for valid syntax.
    Inherits all fields from FileWritten.

    Additionally verifies/does:
    - Path ends with `.md`
    - markdownlint-cli2 is installed
    - markdownlint-cli2 --fix runs successfully to auto format the file
    - markdownlint-cli2 reports no remaining syntax errors after formatting
    - The content field is updated with the formatted content
    """

    @model_validator(mode="before")
    @classmethod
    def _format_markdown_and_fill_content(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        data = dict(data)
        path = data.get("path")
        if not isinstance(path, str) or not path.endswith(".md"):
            return data

        cls._validate_markdown_file(path)
        data["_hdr_content"] = cls._read_file_content(path)
        return data

    def __init__(self, **data):
        super().__init__(**data)
        if not self.path.endswith(".md"):
            raise AssertionError(f"Path '{self.path}' does not end with '.md'")

    @staticmethod
    def _validate_markdown_file(path: str) -> None:
        if not os.path.exists(path):
            raise AssertionError(f"File at {path} does not exist")

        # Check markdownlint-cli2 is installed
        markdownlint_path = shutil.which("markdownlint-cli2")
        if not markdownlint_path:
            raise AssertionError(
                "markdownlint-cli2 is not installed. Please install it with: npm install -g markdownlint-cli2"
            )

        # Run markdownlint-cli2 --fix to auto format the file
        result_fix = subprocess.run(
            ["markdownlint-cli2", "--fix", path],
            capture_output=True,
            text=True,
        )
        if result_fix.returncode != 0:
            raise AssertionError(
                f"markdownlint-cli2 --fix failed for {path}:\n{result_fix.stderr}\n{result_fix.stdout}"
            )

        result_lint = subprocess.run(
            ["markdownlint-cli2", path],
            capture_output=True,
            text=True,
        )
        if result_lint.returncode != 0:
            raise AssertionError(
                f"markdownlint-cli2 found issues in {path} after formatting:\n"
                f"{result_lint.stderr}\n{result_lint.stdout}"
            )


class PythonWorkspaceBuilt(DirectoryCreated):
    """
    Validates a Python workspace is properly configured for linting and type checking.
    Inherits all fields from DirectoryCreated.

    Additionally verifies:
    - ruff is installed (shutil.which)
    - pyright is installed (shutil.which)
    - pyright reports no warnings or errors in the workspace
    - ruff check reports no lint errors in the workspace
    - ruff format runs cleanly in the workspace
    """

    gather_content_on_init = False

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

        # Run ruff format to ensure consistent formatting
        result_ruff_fmt = subprocess.run(
            ["ruff", "format", "."],
            cwd=self.path,
            capture_output=True,
            text=True,
        )
        if result_ruff_fmt.returncode != 0:
            raise AssertionError(
                f"ruff format failed in {self.path}:\n{result_ruff_fmt.stdout}\n{result_ruff_fmt.stderr}"
            )

        if not self.content:
            self.content = self._gather_content(self.path)
            total_files = len(self.content)
            print(f"[Directory] Total files in {self.path}: {total_files}")
