---
name: hdr
description: Harness Done Right - Formalize tasks as Python classes, validate with LLM assertions, and execute tasks incrementally with dependency management. Use this skill whenever the user wants to define formal task structures, validate task completions with LLM checks, build task dependency graphs, or track task execution progress.
---

# Harness Done Right (HDR) Skill

This skill enables you to formalize any task as a Python class hierarchy, validate task completions using LLM-powered assertions, and execute tasks incrementally with proper dependency management.

## Core Concepts

### Task Definition
Every task is defined as a Python class with:
- A constructor that takes parameters representing the task requirements
- LLM assertions that validate the task is correctly completed
- Type checking for all parameters

### Workbench
All completed task instances are stored in a workbench:
- Each task instance has a unique identifier
- Instances can be referenced as dependencies for other tasks using `get("<id>")`
- Each instance can only be used once to avoid circular dependencies
- All instances are persisted to disk for later use

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

# Set the goal task to complete
goal(MyTask)

# Create a task instance and store it in the workbench (Pydantic uses keyword arguments)
create("task-id", MyTask(param1="value1", param2=42, dependency=get("other-task-id")))

# Retrieve a task instance from the workbench
instance = get("task-id")

# Mark the goal as completed using a matching task instance
finish(instance)
```

## Built-in Functions

### `llm_assert(condition: str) -> None`
Validates a condition using LLM. Throws an error with LLM explanation if validation fails.

### `llm_check(predicate: str, value: Any) -> bool`
Runs a predicate check using LLM and returns a boolean result.

### `goal(task_type: Type) -> None`
Sets the target task type that needs to be completed.

### `create(id: str, instance: Any) -> None`
Stores a task instance in the workbench with the given ID. Validates all type constraints and assertions before storing.

### `get(id: str) -> Any`
Retrieves a task instance from the workbench by ID. Marks the instance as consumed so it cannot be reused.

### `finish(instance: Any) -> None`
Marks the goal as completed using the provided instance, which must match the goal task type.

## LLM Integration
- By default, uses OpenRouter API for LLM operations
- Requires `OPENROUTER_API_KEY` environment variable to be set
- Requires `OPENROUTER_MODEL` environment variable to specify the model to use
- Will throw clear error messages if required environment variables are missing

## Usage Workflow

1. **Define Task Structure**: First, formalize all required task types as Python classes with appropriate assertions
2. **Validate Task Definition**: Confirm the task structure with the user before proceeding
3. **Set Goal**: Specify which task type is the final goal to complete
4. **Build Incrementally**: Create dependencies first, then build up to the goal task
5. **Validate Each Step**: Each `create()` call automatically runs all assertions and type checks
6. **Complete Goal**: Use `finish()` with the final goal instance to complete the task

## Error Handling
- Type mismatch errors when passing incorrect types to task constructors
- Instance reuse errors when trying to use the same instance multiple times
- LLM assertion failures with detailed explanations of what went wrong
- Missing environment variable errors with configuration instructions

## Testing
When testing, use the mock LLM mode to avoid actual API calls:
```python
from hdr import mock_llm
mock_llm.enable()
```
