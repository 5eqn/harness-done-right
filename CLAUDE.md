# HDR: Harness Done Right

HDR is a structured task execution framework for **Claude Code**. It allows Claude to formalize tasks as Python classes, where successful instantiation serves as proof of task completion.

## Core Concept
1. **Formalize**: Define a task as a Python class inheriting from `BaseModel`.
2. **Verify**: Use LLM-powered assertions (`verify`) to validate qualitative requirements.
3. **Execute**: Instantiate the class. If validation passes (Score 5/5), the task is complete.

---

## Example: Humanizing Text

### 1. Task Definition
Claude formalizes the requirement into a schema:

```python
from hdr import BaseModel, verify, quote

class HumanizeText(BaseModel):
    original: str
    humanized: str

    def __init__(self, **data):
        super().__init__(**data)
        # LLM-backed assertions
        verify(f"{quote(self.original)} and {quote(self.humanized)} convey the same meaning")
        verify(f"{quote(self.humanized)} reads like natural human-written text")
```

### 2. Execution
Claude runs the script to perform and verify the task:

```python
from hdr import checkout

checkout("current-git-hash")

# Instantiation triggers Pydantic type checks and LLM verification
result = HumanizeText(
    original="AI-generated technical jargon...",
    humanized="A clear, human-friendly explanation..."
)
print("Task Verified:", result)
```

---

## Key Features

*   **Type Safety**: Built on **Pydantic** for runtime type and schema validation.
*   **LLM Assertions**: `verify(condition)` calls Claude to score the result (1-5). Only a score of **5** passes; otherwise, it throws an error with the reasoning.
*   **Commit-Based Caching**: Verification results are cached per git commit to prevent redundant LLM calls.
*   **Prompt Safety**: `quote(obj)` handles any data type safely to prevent prompt injection.
*   **Zero Config**: Use `checkout(commit)` to isolate and prepare the workspace instantly.

---

## Core API

| Function | Description |
| :--- | :--- |
| `checkout(commit)` | Extracts the specified git commit to a temporary workspace. |
| `BaseModel` | Base class for all tasks; provides automatic schema validation. |
| `verify(condition)` | Uses Claude to validate a condition. Fails if score < 5. |
| `quote(obj)` | Safely serializes objects (str, dict, models) for LLM prompts. |

---

## Project Structure

```text
hdr/
└── hdr-skill/              # Project root (Claude Code skill)
    ├── src/
    │   └── hdr/            # Main package
    │       ├── __init__.py # Re-exports from core
    │       ├── core.py     # Core logic & LLM bridge
    │       └── tasks/      # Standard task types
    │           ├── __init__.py
    │           └── std.py  # File, PythonFile, etc.
    ├── tests/              # Unit tests
    │   └── test_core.py
    └── examples/           # Example workflows
        └── introduction_writing/
```

## Quick Setup
```bash
cd hdr-skill
uv venv .venv
source .venv/bin/activate
uv pip install pydantic locache pytest
uv pip install -e .
```

## Usage in Claude Code
When Claude Code is instructed to complete a task with HDR, it should:
1. Define the task class in a `.py` file.
2. Present the schema to the user for approval.
3. Write the implementation logic that instantiates the class.
4. Run the script to finalize and verify the work.
