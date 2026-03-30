---
name: hdr
description: Harness Done Right - Formalize tasks as Python classes, validate with LLM assertions, and execute tasks incrementally with dependency management. Use this skill whenever the user wants to define formal task structures, validate task completions with LLM checks, build task dependency graphs, or track task execution progress.
---

# Harness Done Right (HDR) Skill

This skill enables you to formalize any task as a Python class hierarchy, validate task completions using LLM-powered assertions, and build complex task structures with automatic dependency validation.

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
        llm_assert(f"{self.param1} meets all quality requirements")
        llm_assert(f"{self.param2} is within expected range")
        llm_assert(f"{self.dependency} is correctly used as input")

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

### `llm_assert(condition: str) -> None`
Validates a condition using LLM. Throws an `AssertionError` with LLM reasoning and score if validation fails. Only passes when LLM gives a perfect score of 5/5. Results are automatically cached to avoid duplicate calls.

## Recommended Workflow

### 1. Split task definition and implementation
- **`task.py`**: Define all your task classes here. This file should be immutable once agreed with the user - it represents the formal specification of what needs to be done.
- **`work.py`**: Import task classes from `task.py` and implement the logic to construct the final task instance.

```python
# task.py (immutable specification)
from hdr import BaseModel, llm_assert

class HumanizeText(BaseModel):
    original: str
    humanized: str

    def __init__(self, **data):
        super().__init__(**data)
        llm_assert(f"<a>{self.original}</a> and <b>{self.humanized}</b> conveys the same meaning")
        llm_assert(f"{self.humanized} reads like natural human-written text")
```

```python
# work.py (implementation)
from hdr import save_config
from task import *

# Implement the task
result = HumanizeText(
    original="Text with AI generated content that sounds robotic",
    humanized="Text written in a natural, conversational tone that feels human"
)

print("Task completed successfully!")
```

### 2. Run in virtual environment
Always execute your code in the project's virtual environment to ensure dependencies are correctly installed:
```bash
# Activate venv (from hdr-skill directory)
source .venv/bin/activate

# Run your implementation
python work.py
```

### 3. Validate incrementally
- Build dependencies first, validate they work before moving to higher-level tasks

## Error Handling
- **ValidationError**: Thrown by Pydantic when you pass incorrect types to task constructors
- **AssertionError**: Thrown when an LLM assertion fails, includes reasoning and score
- **EnvironmentError**: Thrown when OpenRouter configuration is missing or invalid
- All errors include clear, actionable instructions for fixing the issue
