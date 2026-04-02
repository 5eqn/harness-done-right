# Standard Tasks Reference

HDR includes a standard library of common task types in `hdr.tasks.std`.

---

## File

**What:** Validates that a file exists (or does not exist) at a given path.

**Why:** Before proceeding with tasks that depend on a file, you want to guarantee it exists. `File` provides a self-verifying proof of presence.

**How to use:**

```python
from hdr.tasks import File

# Verify a file exists
file_task = File(path="README.md", exists=True)

# Verify a file does not exist
missing = File(path="debug.log", exists=False)

# Default is exists=True
default = File(path="README.md")
```

**Fields:**
- `path: str` — path to the file (relative or absolute; relative is preferred for portability)
- `exists: bool = True` — whether the file must exist or must not exist

**Quote behavior:** `quote(File(...))` returns the file's full content wrapped in `<file><path>...</path><content>...</content></file>`.

---

## Directory

**What:** Validates that a directory exists (or does not exist) at a given path. Optionally holds a list of `File` instances for composition with other tasks.

**Why:** When a task operates on a directory, you want to guarantee the directory is present before proceeding. The `files` list allows composition with `File` tasks for joint validation.

**How to use:**

```python
from hdr.tasks import Directory, File

# Verify a directory exists
dir_task = Directory(path="src", exists=True)

# Verify a directory does not exist
missing = Directory(path="/tmp/cache", exists=False)

# Directory with File instances for composition
dir_with_files = Directory(path="src", files=[File(path="main.py"), File(path="utils.py")])

# Nested files
nested = Directory(path="src", files=[File(path="sub/module.py")])
```

**Fields:**
- `path: str` — path to the directory
- `exists: bool = True` — whether the directory must exist or must not exist
- `files: list[File] = []` — list of `File` instances (for composition; Directory does not validate them)

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
