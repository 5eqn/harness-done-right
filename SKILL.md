---
name: hdr-skill
description: Harness Done Right - Formalize tasks as Python classes, validate with LLM assertions, and execute tasks incrementally with dependency management. Use this skill whenever the user wants to define formal task structures, validate task completions with LLM checks, build task dependency graphs, or track task execution progress.
---

# Harness Done Right (HDR) Skill

This skill enables you to formalize any task as a Python class hierarchy, validate task completions using Claude-powered assertions, and build complex task structures with automatic dependency validation.

## Design Philosophy

HDR follows a **stateless, pure-functional design** with three core principles:
1. **Tasks as values**: A completed task is just an instance of a Python class - no hidden state, no workbench, no persistence required
2. **Validation at construction**: All type checks and LLM assertions run automatically when you create a task instance - if it instantiates successfully, it's valid
3. **Caching, not state**: Duplicate LLM calls are automatically cached by condition, so you can rerun your code as many times as you want without extra cost or side effects

## Core Concepts

### Task Definition
Every task is defined as a Python class that inherits from `Task` (extends Pydantic BaseModel for automatic type checking):
- Class fields with `Field(description=...)` define the task's required inputs/outputs with documentation
- The constructor runs LLM assertions using `self.verify()` that validate the task is correctly completed
- All type checks run automatically at instantiation time
- Field descriptions are automatically included in verification prompts

### Stateless Execution
To complete a task:
1. Define your task classes with all required validation logic and field descriptions
2. Construct instances of your task classes directly, passing dependencies as parameters
3. If the final target instance constructs successfully, your task is complete

## Core API

```python
from hdr.tasks.std import Task
from pydantic import Field

# Define a new task type (inherit from Task for automatic type checking and verification)
class MyTask(Task):
    param1: str = Field(description="First parameter description")
    param2: int = Field(description="Second parameter description")
    dependency: OtherTask = Field(description="Dependency task instance")

    def __init__(self, **data):
        super().__init__(**data)
        # LLM assertions to validate task completion
        # self.verify() automatically includes full task state as context
        self.verify("param1 meets all quality requirements")
        self.verify("param2 is within expected range")
        self.verify("dependency is correctly used as input")

# Construct dependency first
dependency = OtherTask(value="some value")

# Construct final task (all validation runs automatically)
result = MyTask(param1="value1", param2=42, dependency=dependency)

# If you reach this line, the task is 100% valid
print("Task completed:", result)
```

## Built-in Functions

### `Task`
Base class for all task types. Extends Pydantic BaseModel to provide automatic runtime type checking for all fields. Supports all Pydantic types including nested models, lists, dicts, etc. Includes built-in `self.verify()` method for LLM-powered validation.

### `self.verify(condition: str) -> None`
Validates a condition against the current task state using Claude. Automatically includes the full pretty-printed task object (with all fields and descriptions) as context. Throws an `AssertionError` with reasoning and score if validation fails. Only passes when Claude gives a perfect score of 5/5. Results are automatically cached by condition to avoid duplicate calls.

No need to manually quote values in conditions - the task context is already included automatically.

Configuration via environment variables:
- `ANTHROPIC_AUTH_TOKEN`: Your Anthropic API key (required)
- `ANTHROPIC_BASE_URL`: The base URL for the API (optional)
- `ANTHROPIC_MODEL`: The model name to use (optional, defaults to claude-4.6-sonnet)

### `quote(obj: Any) -> str`
Safely pretty-print any object for use in prompts. Automatically handles:
- Simple types (strings, numbers, booleans)
- Complex Pydantic models (pretty-printed with class name, fields, and descriptions)
- Lists, dicts, and other Python objects
- Provides indentation and structured formatting for readability

Use `quote()` when you need to explicitly reference objects in custom prompts or error messages.

## Standard Tasks

HDR includes a standard library of common task types in `hdr.tasks.std`:

```python
from hdr.tasks import FileWritten

# Validate a file exists (prefer relative paths)
file_task = FileWritten(path="README.md")
```

Available standard tasks:

| Task | Description |
| :--- | :--- |
| `FileWritten` | Validates a file exists; auto-fills content from disk. |
| `DirectoryCreated` | Validates a directory exists; auto-gathers file content recursively. |
| `PythonWorkspaceBuilt` | Extends `DirectoryCreated`; also validates ruff and pyright pass cleanly. |
| `ConceptDescribed` | Represents a documented concept with LLM quality validation. |

See `docs/tasks/std.md` for full documentation.

## Recommended Workflow

### Agent Execution Steps (any working directory)
Follow this exact step-by-step process for every task:

#### Step 1: Create task specification file
Create a `task.py` file in your current working directory with all task definitions:
- Import `Task` from `hdr.tasks.std` and `Field` from `pydantic`
- Define all required task classes inheriting from `Task`, with proper type annotations and `Field(description=...)` for all fields
- Add all necessary `self.verify()` validations in the `__init__` method - no need to manually quote values, as the full task context is automatically included
- Present this file to the user for approval before proceeding

#### Step 2: Get user confirmation
Once the user approves the `task.py` specification, do not modify this file again for the rest of the task.

#### Step 3: Create implementation file
Create a `work.py` file in the same directory:
- Import all task classes from `task.py`
- Implement the logic to construct the final task instance, building dependencies as needed
- If task is very complicated, you don't have to create the final task instance at the first try

#### Step 4: Run and validate
Execute your code:
```bash
# Run your implementation (timeout 30 minutes)
python work.py
```

Notice:
- If unable to import hdr, ask the user to rerun you inside correct virtual environment.
- If you have attempted to construct the final task instance without placeholder, and the code runs without errors, your task is complete.
- If encountered error, revise your implementation in `work.py` with the error message and rerun the implementation.
- If the code runs successfully, but you haven't created the final task instance, it means the prefix works fine, please continue working on the construction of the final task instance in `work.py`.
- If you think the error reason does not make sense, you think the problem is not in your `work.py` but in `task.py`, request with user to go back to edit `task.py`. User will tell you whether they think the problem is in the task `task.py` or in your work `work.py`.

## Error Handling
- **ValidationError**: Thrown by Pydantic when you pass incorrect types to task constructors
- **AssertionError**: Thrown when a verification fails, includes reasoning and score
- **EnvironmentError**: Thrown when required environment variables (like ANTHROPIC_AUTH_TOKEN) are not set
- All errors include clear, actionable instructions for fixing the issue
