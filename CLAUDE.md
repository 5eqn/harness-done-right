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

## Standard Tasks

| Task | Description |
| :--- | :--- |
| `FileWritten` | Validates a file exists; auto-fills content from disk. |
| `MarkdownFileWritten` | Validates a markdown file exists with valid syntax (extends `FileWritten`). |
| `PythonFileWritten` | Validates a `.py` file exists; checks extension, ruff, and pyright pass cleanly. |
| `DirectoryCreated` | Validates a directory exists; auto-gathers file content recursively. |
| `PythonWorkspaceBuilt` | Extends `DirectoryCreated`; also validates ruff and pyright pass cleanly. |
| `ConceptDescribed` | Represents a documented concept with LLM quality validation. |
| `TaskCreated` | Meta-task for creating well-formed HDR tasks with validated fields and verify examples. |

See `docs/tasks/std.md` for standard tasks, `docs/tasks/coding.md` for Python-related tasks, `docs/tasks/meta.md` for meta-task utilities.

---

## Project Structure

```
hdr-skill/
├── CLAUDE.md              # Project instructions
├── README.md              # User-facing documentation
├── SKILL.md               # Skill metadata for Claude Code
├── docs/
│   └── tasks/
│       ├── std.md         # Standard task documentation
│       ├── coding.md      # Python coding task documentation
│       └── meta.md        # Meta-task utility documentation
├── src/
│   └── hdr/
│       ├── __init__.py    # Public API exports
│       └── tasks/
│           ├── __init__.py
│           ├── std.py     # Core framework + standard tasks
│           ├── coding.py  # Python coding-related tasks
│           └── meta.py    # Meta-task utilities
├── tests/
│   ├── __init__.py
│   ├── test_quote.py       # Quote function tests
│   ├── test_verify.py      # Verify method tests
│   ├── test_std.py         # Standard task tests
│   ├── test_coding.py      # Python coding task tests
│   └── test_meta.py        # Meta-task (TaskCreated) tests
└── examples/
    └── introduction_writing/
        ├── __init__.py
        ├── context.md
        ├── description.md
        ├── task.py         # Task specification
        └── work.py         # Implementation
```
