# Directory and PythonWorkspace Tasks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Directory` and `PythonWorkspace` task classes to `src/hdr/tasks/std.py`, export them from `__init__.py`, and add tests.

**Architecture:** `Directory` mirrors `File` but for directory paths with an optional `files` list. `PythonWorkspace` inherits from `Directory` and adds tool checks via `shutil.which` and `subprocess.run`.

**Tech Stack:** Python, Pydantic (BaseModel), subprocess, shutil, tempfile

---

## File Structure

- Modify: `src/hdr/tasks/std.py` — add Directory and PythonWorkspace classes
- Modify: `src/hdr/tasks/__init__.py` — export Directory and PythonWorkspace
- Create: `tests/test_std.py` — Directory tests

---

## Task 1: Add Directory class to std.py

**Files:**
- Modify: `src/hdr/tasks/std.py:1-44`

- [ ] **Step 1: Read current std.py**

```python
# src/hdr/tasks/std.py (current, lines 1-44)
"""
Standard task types for common use cases.
...
class File(BaseModel):
    path: str
    exists: bool = True

    def __init__(self, **data):
        super().__init__(**data)
        file_exists = os.path.exists(self.path)
        if self.exists and not file_exists:
            raise AssertionError(f"File at {self.path} does not exist")
        if not self.exists and file_exists:
            raise AssertionError(f"File at {self.path} should not exist")
```

- [ ] **Step 2: Add Directory class after File**

Append this to `src/hdr/tasks/std.py`:

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add src/hdr/tasks/std.py
git commit -m "feat: add Directory task class"
```

---

## Task 2: Add PythonWorkspace class to std.py

**Files:**
- Modify: `src/hdr/tasks/std.py` (already in working tree)

- [ ] **Step 1: Read std.py to find insertion point**

The file now ends with Directory's `model_dump_json`. Import `shutil` and `subprocess` at the top if not already present.

- [ ] **Step 2: Add import statements**

At the top of `src/hdr/tasks/std.py`, update the imports from:

```python
import os
from hdr.core import BaseModel
```

to:

```python
import os
import shutil
import subprocess
from hdr.core import BaseModel
```

- [ ] **Step 3: Add PythonWorkspace class after Directory**

Append to `src/hdr/tasks/std.py`:

```python
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
```

- [ ] **Step 4: Commit**

```bash
git add src/hdr/tasks/std.py
git commit -m "feat: add PythonWorkspace task class"
```

---

## Task 3: Export Directory and PythonWorkspace from tasks/__init__.py

**Files:**
- Modify: `src/hdr/tasks/__init__.py`

- [ ] **Step 1: Update exports**

```python
from hdr.tasks.std import Directory, File, PythonWorkspace

__all__ = ["Directory", "File", "PythonWorkspace"]
```

- [ ] **Step 2: Commit**

```bash
git add src/hdr/tasks/__init__.py
git commit -m "feat: export Directory and PythonWorkspace"
```

---

## Task 4: Add tests for Directory

**Files:**
- Create: `tests/test_std.py`

- [ ] **Step 1: Write failing tests**

```python
"""
Tests for Directory task class.
Uses a fixed temp workspace created via tempfile.mkdtemp.
"""

import os
import tempfile
import pytest
from hdr.tasks.std import Directory

class TestDirectory:
    """Tests for Directory task class."""

    def test_directory_exists(self):
        """Test Directory validation passes when directory exists and exists=True"""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Directory(path=tmpdir, exists=True)
            assert d.path == tmpdir
            assert d.exists is True

    def test_directory_not_exists(self):
        """Test Directory validation fails when directory does not exist and exists=True"""
        with pytest.raises(AssertionError, match="does not exist"):
            Directory(path="/nonexistent/path/12345", exists=True)

    def test_directory_exists_false_when_missing(self):
        """Test Directory validation passes when directory does not exist and exists=False"""
        d = Directory(path="/nonexistent/path/12345", exists=False)
        assert d.path == "/nonexistent/path/12345"
        assert d.exists is False

    def test_directory_exists_false_when_present(self):
        """Test Directory validation fails when directory exists but exists=False"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(AssertionError, match="should not exist"):
                Directory(path=tmpdir, exists=False)

    def test_directory_default_exists_true(self):
        """Test Directory defaults to exists=True"""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Directory(path=tmpdir)
            assert d.exists is True

    def test_directory_with_files(self):
        """Test Directory validates that all listed files exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file inside the directory
            subfile = os.path.join(tmpdir, "sample.py")
            with open(subfile, "w") as f:
                f.write("# sample")
            d = Directory(path=tmpdir, files=["sample.py"])
            assert d.path == tmpdir
            assert d.files == ["sample.py"]

    def test_directory_with_missing_file(self):
        """Test Directory raises when listed file does not exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(AssertionError, match="does not exist"):
                Directory(path=tmpdir, files=["nonexistent.py"])

    def test_directory_nested_files(self):
        """Test Directory validates nested files (relative path)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "sub")
            os.makedirs(nested_dir)
            nested_file = os.path.join(nested_dir, "nested.py")
            with open(nested_file, "w") as f:
                f.write("# nested")
            d = Directory(path=tmpdir, files=["sub/nested.py"])
            assert d.files == ["sub/nested.py"]

    def test_directory_multiple_files(self):
        """Test Directory validates multiple files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_a = os.path.join(tmpdir, "a.py")
            file_b = os.path.join(tmpdir, "b.py")
            with open(file_a, "w") as f:
                f.write("# a")
            with open(file_b, "w") as f:
                f.write("# b")
            d = Directory(path=tmpdir, files=["a.py", "b.py"])
            assert d.files == ["a.py", "b.py"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: Run tests to verify they fail (Directory not yet exported)**

```bash
cd /Users/wujimi/hdr-skill && python -m pytest tests/test_std.py -v 2>&1
```

Expected: import error or test failure

- [ ] **Step 3: Implement the fix (update __init__.py if needed, ensure Task 3 was done)**

Actually Task 3 already updated exports. Run tests again:

```bash
python -m pytest tests/test_std.py -v
```

Expected: PASS for all tests

- [ ] **Step 4: Commit**

```bash
git add tests/test_std.py
git commit -m "test: add Directory task tests"
```
