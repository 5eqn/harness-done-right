# Standard Tasks

Standard task types in `hdr.tasks.std` for common use cases. All tasks use relative paths by preference, as they are more portable and make projects easier to share and version control.

## Base

### Task

Base class for all HDR tasks. Extends Pydantic `BaseModel` to provide automatic runtime type checking for all fields. Includes a built-in `self.verify()` method for LLM-powered validation.

**Key methods:**

- `self.verify(condition: str) -> None` — Validates a condition against the current task state using Claude. Automatically includes the full pretty-printed task object as context. Throws `AssertionError` with reasoning and score if validation fails. Only passes at score 5/5. Results are automatically cached by condition.

## File Tasks

### FileWritten

Validates that a file exists at the given path and optionally reads its content.

**Fields:**
- `path: str` — Path to the file (relative paths recommended).
- `content: str` — Content of the file; auto-filled from the actual file if not provided.

**Validates:**
- File exists at `path` (`os.path.exists`).
- File is readable.

**Example:**
```python
from hdr import FileWritten

file = FileWritten(path="README.md")
print(file.content)  # auto-filled from disk
```

## Directory Tasks

### DirectoryCreated

Validates that a directory exists and gathers its file content recursively, respecting `.gitignore` patterns.

**Fields:**
- `path: str` — Path to the directory.
- `content: list[FileWritten]` — List of FileWritten objects representing files in the directory; auto-filled if not provided.

**Validates:**
- Directory exists at `path` (`os.path.isdir`).
- Recursively gathers file content, skipping entries matching `.gitignore` patterns.

**Example:**
```python
from hdr import DirectoryCreated

directory = DirectoryCreated(path="src")
for f in directory.content:
    print(f.path, len(f.content))
```

### PythonWorkspaceBuilt

Extends `DirectoryCreated` — validates a Python workspace is properly configured for linting and type checking.

**Inherits:** All fields from `DirectoryCreated`.

**Validates (in addition to directory existence):**
- `ruff` is installed (`shutil.which("ruff")`).
- `pyright` is installed (`shutil.which("pyright")`).
- `pyright --outputjson` reports zero errors and zero warnings.
- `ruff check .` reports no lint errors.

**Example:**
```python
from hdr import PythonWorkspaceBuilt

workspace = PythonWorkspaceBuilt(path="src/my_project")
```

## Concept Tasks

### ConceptDescribed

Represents a documented concept within a context, with LLM validation of description quality.

**Fields:**
- `context: FileWritten` — Markdown file explaining the parent context.
- `name: str` — Name of the concept.
- `description: FileWritten` — Markdown file containing the concept description.

**Validates (via `self.verify`):**
- The description is written for readers who understand context but do not yet know the concept name.
- The concept name represents exactly one atomic idea.
- The description contains no time-sensitive terms without specifying an exact version or date.
- The description identifies a broader category and a distinguishing property.
- A reader familiar with context can classify concrete instances of the concept.

**Example:**
```python
from hdr import ConceptDescribed, FileWritten

concept = ConceptDescribed(
    context=FileWritten(path="context.md"),
    name="Connection Pooling",
    description=FileWritten(path="description.md"),
)
```
