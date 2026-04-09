"""
Coding tasks for Python-related validation.

These tasks validate Python files and workspaces, using tools like ruff and pyright.
"""

import shutil
import subprocess

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

    def __init__(self, **data):
        super().__init__(**data)
        if not self.path.endswith(".py"):
            raise AssertionError(f"Path '{self.path}' does not end with '.py'")

        # Check pyright is installed
        pyright_path = shutil.which("pyright")
        if not pyright_path:
            raise AssertionError(
                "pyright is not installed. Please install it with: pip install pyright"
            )

        # Run pyright on the single file
        result_pyright = subprocess.run(
            ["pyright", "--outputjson", self.path],
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
                    f"in {self.path}. Run 'pyright {self.path}' for details."
                )
        except (json.JSONDecodeError, KeyError):
            raise AssertionError(
                f"pyright produced unexpected output for {self.path} "
                f"(exit code {result_pyright.returncode}). Could not verify. Run 'pyright {self.path}' for details."
            )

        # Run ruff check on the single file
        result_ruff = subprocess.run(
            ["ruff", "check", self.path],
            capture_output=True,
            text=True,
        )
        if result_ruff.returncode != 0:
            raise AssertionError(
                f"ruff found lint errors in {self.path}:\n{result_ruff.stdout}\n{result_ruff.stderr}"
            )

        # Run ruff format on the single file
        result_ruff_fmt = subprocess.run(
            ["ruff", "format", "--check", self.path],
            capture_output=True,
            text=True,
        )
        if result_ruff_fmt.returncode != 0:
            raise AssertionError(
                f"ruff format check failed for {self.path}:\n{result_ruff_fmt.stdout}\n{result_ruff_fmt.stderr}"
            )


class MarkdownFileWritten(FileWritten):
    """
    Validates that a markdown file exists at the given path.
    Inherits all fields from FileWritten.

    Additionally verifies:
    - Path ends with `.md`
    - markdownlint-cli2 reports no syntax errors
    """

    def __init__(self, **data):
        super().__init__(**data)
        if not self.path.endswith(".md"):
            raise AssertionError(f"Path '{self.path}' does not end with '.md'")
        result = subprocess.run(
            ["markdownlint-cli2", self.path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"markdownlint-cli2 found issues in {self.path}:\n{result.stderr}\n{result.stdout}"
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
