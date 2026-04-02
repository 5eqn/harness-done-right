# Standard Tasks Reference

HDR includes a standard library of common task types in `hdr.tasks.std`.

---

## File

**What:** Validates that a file exists at a given path using `os.path.exists()`. The `content` field is auto-filled from the actual file content if not specified.

**Why:** Before proceeding with tasks that depend on a file, you want to guarantee it exists. `File` provides a self-verifying proof of presence.

**How to use:**

```python
from hdr.tasks import File

# Verify a file exists and auto-fill content
file_task = File(path="README.md")

# Provide explicit content
custom = File(path="output.txt", content="my content")
```

**Fields:**
- `path: str` — path to the file (relative or absolute; relative is preferred for portability)
- `content: str = ""` — the file content (auto-filled from file if not specified)

---

## Directory

**What:** Validates that a directory exists at a given path using `os.path.isdir()`. The `content` field is a list of `File` objects auto-filled from the actual directory content if not specified. Content is gathered recursively, respecting `.gitignore` patterns, and the total file count is logged.

**Why:** When a task operates on a directory, you want to guarantee the directory is present before proceeding. The auto-filled `content` allows inspection of directory contents as a list of `File` objects.

**How to use:**

```python
from hdr.tasks import Directory, File

# Verify a directory exists and auto-fill content
dir_task = Directory(path="src")

# Provide explicit content as list[File]
custom = Directory(path="output", content=[File(path="a.txt", content="my content")])
```

**Fields:**
- `path: str` — path to the directory
- `content: list[File] = []` — the directory content as list of File objects (auto-filled from directory if not specified, respects .gitignore, recurses)

---

## PythonWorkspace

**What:** Validates a Python workspace has proper linting and type-checking configured. Inherits from `Directory` — the directory must exist. Additionally verifies `ruff` and `pyright` are installed and produce no errors or warnings.

**Why:** Before running Python development tasks in a workspace, you want to guarantee the tooling is set up correctly and the code passes all linting and type checks. If tools are missing, the error message instructs the caller to install them.

**How to use:**

```python
from hdr.tasks import PythonWorkspace

# Verify a Python workspace is properly configured
workspace = PythonWorkspace(path=".")

# With a subdirectory
workspace = PythonWorkspace(path="src")
```

**What it checks (in order):**
1. Directory exists (inherited from `Directory`)
2. `ruff` is installed (`shutil.which`)
3. `pyright` is installed (`shutil.which`)
4. `pyright --outputjson` in the workspace produces zero errors and zero warnings
5. `ruff check .` in the workspace returns exit code 0

**If tools are missing:** Raises `AssertionError` with an install instruction, e.g.:
```
ruff is not installed. Please install it with: pip install ruff
```

**If linting/type-checking fails:** Raises `AssertionError` with details:
```
pyright found 2 error(s) and 0 warning(s) in /path/to/workspace. Run 'pyright' for details.
```
