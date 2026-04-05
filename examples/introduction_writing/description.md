# HDR (Harness Done Right)

HDR is a **structured task execution framework** for AI coding agents. It lets you define any task as a Python class that combines input fields, transformation logic, and LLM-powered verification into a single construct—where **successful class instantiation is the proof of task completion**.

## Core Idea

You define a task by inheriting from `Task` and adding fields with descriptions:

```python
from hdr.tasks.std import Task
from pydantic import Field

class HumanizeText(Task):
    original: str = Field(description="Original AI-generated text")
    humanized: str = Field(description="Humanized version")

    def __init__(self, **data):
        super().__init__(**data)
        self.verify("original and humanized convey the same meaning")
        self.verify("humanized reads like natural human-written text")
```

When you instantiate the class with actual values, Pydantic validates the types automatically, then the `self.verify()` calls invoke an LLM to judge whether the qualitative requirements are met. Only a perfect score (5/5) passes; anything less raises an error with reasoning.

This separates HDR from other approaches: **verification happens at construction time, as part of the object graph, not as a separate process or conversation turn.** A completed task is just a valid object instance—no callbacks, no separate test runners, no "did it work?" uncertainty.

## What Constitutes an HDR Task

An HDR task is any Python class that inherits from `Task` and whose completion is gatekept by `self.verify()` calls that run during instantiation.

Concretely, a class is an HDR task when:
1. It inherits from `Task`
2. Its `__init__` directly contains at least one unconditional `self.verify()` call—not a call to a helper method, not a conditional call, but an explicit `self.verify()` in the `__init__` body itself
3. Reaching the end of `__init__` without an `AssertionError` means the task is complete

Inherited `verify()` calls from parent classes do not count unless the child class also directly calls `self.verify()` in its own `__init__`. Decorator-wrapped `__init__` methods are evaluated as they appear. A call inside an `if True:` block is considered unconditional. Calls inside context managers, loops, or `__new__` do not count. Metaclass-created `__init__` methods are evaluated as written.

A `Task` subclass with no `verify()` calls is not an HDR task. A class that calls verify() only inside helper methods, context managers, loops, or only conditionally is not an HDR task.

## What HDR Feels Like

HDR occupies a specific niche: **agents working on complex, multi-step tasks where qualitative judgment matters as much as type correctness.**

It is not a testing library (pytest does that better). It is not a prompt engineering tool (Conversation does that better). Instead, it is a way to say: "Here is what this subtask requires, here is what counts as done, and I want the framework to enforce both automatically."

The distinguishing property is the **construction-time verification pattern**: you build the final object graph by composing smaller task objects, and if the whole thing instantiates without raising an `AssertionError`, you have a completed task with LLM-verified quality gates baked in.
