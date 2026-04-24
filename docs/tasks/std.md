# Standard Contracts

Standard contract types in `hdr.contracts.std` for common use cases. All contracts use relative paths by preference, as they are more portable and make projects easier to share and version control.

## Base

### BaseContract

Base class for all HDR contracts. Extends Pydantic `BaseModel` to provide automatic runtime type checking for all fields. Includes a built-in `self.llm_verify()` method for LLM-powered validation.

**Key methods:**

- `self.llm_verify(condition: str) -> None` — Validates a condition against the current contract state using Claude. Automatically includes the full pretty-printed contract object as context. On first use, HDR creates `~/.hdr/config.yaml` if it does not exist; you must fill in `anthropic_auth_token` there before verification can run. The config also supports `verify_cache_dir`, which defaults to `/tmp/claude/hdr_verify_cache`. On success, logs a one-line message with the actual score and a trimmed condition preview. Throws `AssertionError` with reasoning and score if validation fails. Only passes at score 5/5 by default. Results are automatically cached by condition.

## File Contracts

### File

Validates that a file exists at the given path and enforces its content when provided.

**Fields:**
- `path: str` — Path to the file (relative paths recommended).
- `content: str` — Content of the file; auto-filled from disk when omitted, or validated against disk when provided.

**Validates:**
- File exists at `path` (`os.path.exists`).
- File is readable.
- If `content` is provided, it exactly matches the file on disk.
- Otherwise, `content` is populated from the actual file content and frozen after initialization.

**Example:**
```python
from hdr.contracts.std import File

file = File(path="config.json", content='{"debug": false}\n')
print(file.content)  # auto-filled from disk
```

## Directory Contracts

### Directory

Validates that a directory exists and that a manually supplied file manifest matches the directory's immediate file tree after applying that directory's `.gitignore`.

**Fields:**
- `path: str` — Path to the directory.
- `content: list[File]` — List of `File` objects that must be manually assigned and whose paths must exactly match the directory's immediate files after `.gitignore` filtering.

**Validates:**
- Directory exists at `path` (`os.path.isdir`).
- `content` is manually provided.
- The provided file paths match the real directory exactly, with no missing or extra files after applying the current directory's `.gitignore`.
- Nested directories are not traversed; each directory should be validated separately.

**Example:**
```python
from hdr.contracts.std import Directory, File

directory = Directory(
    path="src",
    content=[
        File(path="src/main.py"),
        File(path="src/utils.py"),
    ],
)
for f in directory.content:
    print(f.path, len(f.content))
```
