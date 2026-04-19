# Coding Contracts

Coding contract types in `hdr.tasks.coding` for programming-related validation. These contracts validate code files (Python, Markdown) using tools like `ruff`, `pyright`, and `markdownlint-cli2`.

## PythonFile

Validates that a Python file exists and passes linting and type checking.

**Inherits:** All fields from `File`.

**Validates (in addition to file existence):**
- Path ends with `.py` (without LLM).
- `pyright` is installed (`shutil.which("pyright")`).
- `pyright --outputjson <file>` reports zero errors and zero warnings for the file.
- `ruff check <file>` reports no lint errors.
- `ruff format <file>` runs successfully and may update the file. The `content` field is populated after formatting, so it reflects the final disk content.

**Example:**
```python
from hdr.tasks.coding import PythonFile

py_file = PythonFile(path="src/my_project/main.py")
print(py_file.content)  # auto-filled from disk
```

## MarkdownFile

Validates that a markdown file exists at the given path, auto-formats it, and checks for valid syntax.

**Inherits:** All fields from `File`.

**Validates/does (in addition to file existence):**
- Path ends with `.md`.
- `markdownlint-cli2` is installed.
- `markdownlint-cli2 --fix <file>` runs successfully and may update the file.
- `markdownlint-cli2 <file>` reports no remaining issues after formatting.
- The `content` field is updated with the formatted content from disk (cannot be manually assigned).

**Example:**
```python
from hdr.tasks.coding import MarkdownFile

md_file = MarkdownFile(path="README.md")
print(md_file.content)  # auto-filled from disk (now formatted!)
```

## PythonWorkspace

Extends `Directory` — validates a Python workspace is properly configured for linting and type checking.

**Inherits:** All fields from `Directory`.

**Validates (in addition to directory existence):**
- `ruff` is installed (`shutil.which("ruff")`).
- `pyright` is installed (`shutil.which("pyright")`).
- `pyright --outputjson` reports zero errors and zero warnings.
- `ruff check .` reports no lint errors.
- `ruff format .` runs successfully and may update files. Directory content is gathered after formatting, so it reflects the final disk content.

**Example:**
```python
from hdr.tasks.coding import PythonWorkspace

workspace = PythonWorkspace(path="src/my_project")
```
