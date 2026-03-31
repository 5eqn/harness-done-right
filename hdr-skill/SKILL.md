---
name: hdr
description: Harness Done Right - Formalize tasks as Python classes, validate with LLM assertions, and execute tasks incrementally with dependency management. Use this skill whenever the user wants to define formal task structures, validate task completions with LLM checks, build task dependency graphs, or track task execution progress.
---

# Harness Done Right (HDR) Skill

This skill enables you to formalize any task as a Python class hierarchy, validate task completions using Claude Code-powered assertions, and build complex task structures with automatic dependency validation.

## Design Philosophy

HDR follows a **stateless, pure-functional design** with three core principles:
1. **Tasks as values**: A completed task is just an instance of a Python class - no hidden state, no workbench, no persistence required
2. **Validation at construction**: All type checks and LLM assertions run automatically when you create a task instance - if it instantiates successfully, it's valid
3. **Caching, not state**: Duplicate LLM calls are automatically cached, so you can rerun your code as many times as you want without extra cost or side effects

No global state, no magic ID references, no persistence layer - your code is the single source of truth.

## Core Concepts

### Task Definition
Every task is defined as a Python class that inherits from `BaseModel` (for automatic type checking):
- Class fields define the task's required inputs/outputs
- The constructor runs LLM assertions that validate the task is correctly completed
- All type checks run automatically at instantiation time

### Stateless Execution
To complete a task:
1. Define your task classes with all required validation logic
2. Construct instances of your task classes directly, passing dependencies as parameters
3. If the final target instance constructs successfully, your task is complete

## Core API

```python
from hdr import *

# Define a new task type (inherit from BaseModel for automatic type checking)
class MyTask(BaseModel):
    param1: str
    param2: int
    dependency: OtherTask

    def __init__(self, **data):
        super().__init__(**data)
        # LLM assertions to validate task completion
        verify(f"{self.param1} meets all quality requirements")
        verify(f"{self.param2} is within expected range")
        verify(f"{self.dependency} is correctly used as input")

# Construct dependency first
dependency = OtherTask(value="some value")

# Construct final task (all validation runs automatically)
result = MyTask(param1="value1", param2=42, dependency=dependency)

# If you reach this line, the task is 100% valid
print("Task completed:", result)
```

## Built-in Functions

### `BaseModel`
Base class for all task types. Provides automatic runtime type checking for all fields. Supports all Pydantic types including nested models, lists, dicts, etc.

### `verify(condition: str) -> None`
Validates a condition using Claude Code. Throws an `AssertionError` with reasoning and score if validation fails. Only passes when Claude Code gives a perfect score of 5/5. Results are automatically cached to avoid duplicate calls.

### `quote(obj: Any) -> str`
Safely quote any object for use in `verify` conditions, preventing prompt injection attacks. Automatically handles:
- Simple types (strings, numbers, booleans)
- Complex Pydantic models (dumps to pretty JSON)
- Lists, dicts, and other JSON-serializable objects
- All quoted content is wrapped in `<quote>` tags and treated as literal text by the LLM

Always use `quote()` when embedding values or objects in `verify` conditions.

## Recommended Workflow

### Agent Execution Steps (any working directory)
Follow this exact step-by-step process for every task:

#### Step 1: Create task specification file
Create a `task.py` file in your current working directory with all task definitions:
- Import `BaseModel`, `verify`, and `quote` from `hdr`
- Define all required task classes with proper type annotations
- Add all necessary `verify` validations in the `__init__` method, using `quote()` for all embedded values/objects
- Present this file to the user for approval before proceeding

#### Step 2: Get user confirmation
Once the user approves the `task.py` specification, do not modify this file again for the rest of the task.

#### Step 3: Create implementation file
Create a `work.py` file in the same directory:
- Import all task classes from `task.py`
- Implement the logic to construct the final task instance, building dependencies as needed
- If task is very complicated, you don't have to create the final task instance at the first try

#### Step 4: Run and validate
Execute your code in the HDR virtual environment:
```bash
# Activate HDR venv
source /path/to/hdr/hdr-skill/.venv/bin/activate

# Run your implementation
# This may run for a very long time, do not set timeout
python work.py
```
- If you have attempted to construct the final task instance without placeholder, and the code runs without errors, your task is complete.
- If encountered error, revise your implementation in `work.py` with the error message and rerun the implementation.
- If the code runs successfully, but you haven't created the final task instance, it means the prefix works fine, please continue working on the construction of the final task instance in `work.py`.
- If you think the error reason does not make sense, you think the problem is not in your `work.py` but in `task.py`, request with user to go back to edit `task.py`. User will tell you whether they think the problem is in the task `task.py` or in your work `work.py`.

## Error Handling
- **ValidationError**: Thrown by Pydantic when you pass incorrect types to task constructors
- **AssertionError**: Thrown when a verification fails, includes reasoning and score
- All errors include clear, actionable instructions for fixing the issue
