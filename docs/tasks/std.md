# Standard Tasks

Standard task types in `hdr.tasks.std` for common use cases. All tasks use relative paths by preference, as they are more portable and make projects easier to share and version control.

## Base

### Task

Base class for all HDR tasks. Extends Pydantic `BaseModel` to provide automatic runtime type checking for all fields. Includes a built-in `self.verify()` method for LLM-powered validation.

**Key methods:**

- `self.verify(condition: str) -> None` — Validates a condition against the current task state using Claude. Automatically includes the full pretty-printed task object as context. On first use, HDR creates `~/.hdr/config.yaml` if it does not exist; you must fill in `anthropic_auth_token` there before verification can run. Throws `AssertionError` with reasoning and score if validation fails. Only passes at score 5/5. Results are automatically cached by condition.

## File Tasks

### FileWritten

Validates that a file exists at the given path and optionally reads its content.

**Fields:**
- `path: str` — Path to the file (relative paths recommended).
- `content: str` — Content of the file; auto-filled from disk (cannot be manually assigned).

**Validates:**
- File exists at `path` (`os.path.exists`).
- File is readable.
- Content is automatically populated from the actual file content.

**Example:**
```python
from hdr import FileWritten

file = FileWritten(path="config.json")
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
