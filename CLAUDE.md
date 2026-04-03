# HDR: Harness Done Right

HDR is a structured task execution framework for **Claude Code**. It allows Claude to formalize tasks as Python classes, where successful instantiation serves as proof of task completion.

## Core Concept
1. **Formalize**: Define a task as a Python class inheriting from `Task`.
2. **Verify**: Use LLM-powered assertions (`self.verify`) to validate qualitative requirements.
3. **Execute**: Instantiate the class. If validation passes (Score 5/5), the task is complete.

---

## Example: Humanizing Text

### 1. Task Definition
Claude formalizes the requirement into a schema:

```python
from hdr.tasks.std import Task
from pydantic import Field

class HumanizeText(Task):
    original: str = Field(description="Original AI-generated text to humanize")
    humanized: str = Field(description="Human-friendly version of the text")

    def __init__(self, **data):
        super().__init__(**data)
        # LLM-backed assertions
        # self.verify automatically includes full task context
        self.verify("original and humanized convey the same meaning")
        self.verify("humanized reads like natural human-written text")
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
*   **LLM Assertions**: `self.verify(condition)` calls Claude to score the result (1-5). Only a score of **5** passes; otherwise, it throws an error with the reasoning. Automatically includes full task state as context.
*   **Descriptive Fields**: Use `Field(description=...)` to document task fields, which are automatically included in verification prompts.
*   **Message-Based Caching**: Verification results are cached by condition to prevent redundant LLM calls.
*   **Prompt Safety**: `quote(obj)` handles any data type safely with pretty-printing for use in prompts.
*   **Environment Configuration**: API key and model can be configured via environment variables.

---

## Core API

| Function | Description |
| :--- | :--- |
| `Task` | Base class for all tasks; provides automatic schema validation and built-in verification. |
| `self.verify(condition)` | Uses Claude to validate a condition against the current task state. Fails if score < 5. Requires `ANTHROPIC_AUTH_TOKEN` environment variable. |
| `quote(obj)` | Safely pretty-prints objects (str, dict, models) for use in LLM prompts. |

---

## Project Structure

```text
hdr-skill/              # Project root (Claude Code skill)
├── README.md           # User-facing documentation
├── SKILL.md            # Skill metadata for Claude Code
├── src/
│   └── hdr/            # Main package
│       ├── __init__.py # Public API exports
│       └── tasks/      # Task implementations
│           ├── __init__.py
│           └── std.py  # Core framework, standard tasks (File, Directory, PythonWorkspace, etc.)
├── tests/              # Unit tests
│   ├── test_core.py
│   └── test_std.py
├── docs/
│   └── tasks/          # Task documentation
│       └── std.md      # File, Directory, PythonWorkspace reference
└── examples/           # Example workflows
    └── introduction_writing/
```
