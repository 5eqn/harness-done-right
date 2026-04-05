# Coding Tasks

Coding task types in `hdr.tasks.coding` for Python-related validation. These tasks use tools like `ruff` and `pyright` to validate Python files and workspaces.

## PythonFileWritten

Validates that a Python file exists and passes linting and type checking.

**Inherits:** All fields from `FileWritten`.

**Validates (in addition to file existence):**
- Path ends with `.py` (without LLM).
- `pyright` is installed (`shutil.which("pyright")`).
- `pyright --outputjson <file>` reports zero errors and zero warnings for the file.
- `ruff check <file>` reports no lint errors.
- `ruff format --check <file>` reports clean formatting.

**Example:**
```python
from hdr import PythonFileWritten

py_file = PythonFileWritten(path="src/my_project/main.py")
print(py_file.content)  # auto-filled from disk
```

## PythonWorkspaceBuilt

Extends `DirectoryCreated` — validates a Python workspace is properly configured for linting and type checking.

**Inherits:** All fields from `DirectoryCreated`.

**Validates (in addition to directory existence):**
- `ruff` is installed (`shutil.which("ruff")`).
- `pyright` is installed (`shutil.which("pyright")`).
- `pyright --outputjson` reports zero errors and zero warnings.
- `ruff check .` reports no lint errors.
- `ruff format .` runs cleanly.

**Example:**
```python
from hdr import PythonWorkspaceBuilt

workspace = PythonWorkspaceBuilt(path="src/my_project")
```
