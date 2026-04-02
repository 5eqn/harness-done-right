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
# Set up environment variables:
# export ANTHROPIC_AUTH_TOKEN="your-api-key"
# export ANTHROPIC_MODEL="claude-4.6-sonnet"  # optional, defaults to claude-4.6-sonnet

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
*   **Message-Based Caching**: Verification results are cached by condition to prevent redundant LLM calls.
*   **Prompt Safety**: `quote(obj)` handles any data type safely to prevent prompt injection.
*   **Environment Configuration**: API key and model can be configured via environment variables.

---

## Core API

| Function | Description |
| :--- | :--- |
| `BaseModel` | Base class for all tasks; provides automatic schema validation. |
| `verify(condition)` | Uses Claude to validate a condition. Fails if score < 5. Requires `ANTHROPIC_AUTH_TOKEN` environment variable. |
| `quote(obj)` | Safely serializes objects (str, dict, models) for LLM prompts. |

---

## Project Structure

```text
hdr-skill/              # Project root (Claude Code skill)
├── README.md           # User-facing documentation
├── SKILL.md            # Skill metadata for Claude Code
├── src/
│   └── hdr/            # Main package
│       ├── __init__.py # Re-exports from core
│       ├── core.py     # Core logic & LLM bridge
│       └── tasks/      # Standard task types
│           ├── __init__.py
│           └── std.py  # File, Directory, PythonWorkspace
├── tests/              # Unit tests
│   ├── test_core.py
│   └── test_std.py
├── docs/
│   ├── tasks/          # Task documentation
│   │   └── std.md      # File, Directory, PythonWorkspace reference
│   └── superpowers/    # Planning artifacts (from superpowers skill)
│       └── plans/
└── examples/           # Example workflows
    └── introduction_writing/
```
