---
name: hdr-define
description: Define a new HDR task from a plain-language task description by writing a content-reflecting runner file that instantiates `hdr.tasks.meta.TaskCreated` and generates a content-reflecting task file. Use when the user asks for `/hdr-define`, wants to formalize a new task with the installed `hdr` library, or wants a task type created from requirements instead of writing the class by hand.
---

# HDR Define

## Overview

Turn a natural-language request into a proper HDR task definition.
Follow HDR's core philosophy: tasks are values, validation happens at construction time, and the generated task file should be specific, typed, and self-validating.

## HDR Model

Carry these library rules into every task:

- A completed task is an instance of a Python class.
- `Task` extends Pydantic, so field types validate at instantiation.
- `self.verify(condition)` evaluates semantic conditions against the full task state.
- Verification results are cached by condition, so reruns are cheap and stateless.
- `TaskCreated` creates a concrete task class, writes the generated Python file, validates the file with `PythonFileWritten`, and embeds positive and negative examples into the generated `verify(...)` strings.

## TaskCreated Interface

Write against this interface directly:

```python
TaskCreated(
    class_name: str,
    parent_class: str = "Task",
    docstring: str,
    imports: list[str] = [
        "from pydantic import Field",
        "from hdr.tasks.std import Task",
    ],
    fields: list[FieldSpec],
    programmatic_checks: list[str] = [],
    verifies: list[VerifySpec],
)
```

Use these companion models:

```python
FieldSpec(
    name: str,
    type_annotation: str,
    description: str,
    default: str | None = None,
)

VerifySpec(
    condition: str,
    positive_example: dict[str, Any],
    negative_example: dict[str, Any],
)
```

`TaskCreated` derives the generated file name from `class_name` by converting `PascalCase` to `snake_case.py`.

## Workflow

1. Read this skill file and work from the interface described here.
2. Use `.` as the working directory by default.
3. Create a content-reflecting runner file in that directory.
4. Build one `TaskCreated(...)` instance inside that runner file.
5. Let `TaskCreated` generate a specific snake_case task-definition file derived from the class name.
6. Run `python <content_reflecting_runner>.py` from that directory.
7. Ask the user to run the agent inside the Python environment where `hdr` is installed.

## Naming Rules

- Choose a task class name in `PascalCase` that states the completed artifact or state.
- Prefer a narrow noun phrase with a completion verb or adjective, such as `ConceptDescribed`, `ApiContractWritten`, `FeaturePlanDrafted`, or `LandingPageCopyWritten`.
- Prefer content-reflecting names over library-reflecting names.
- Since `TaskCreated` derives the file name from `class_name`, choose the class name so the snake_case file name is also specific and readable.
- Use a generated file name that reflects content, such as `concept_described.py` or `feature_plan_drafted.py`.
- Name the runner file after the task-definition activity, such as `define_concept_described.py`, `build_feature_plan_drafted_task.py`, or `create_api_contract_written.py`.

## Authoring Rules

- Build the task by creating one `TaskCreated(...)` instance in the content-reflecting runner file.
- Use `FieldSpec` and `VerifySpec` explicitly instead of raw unstructured dicts when practical.
- Give every field a concrete description.
- Prefer relative paths.
- Keep programmatic checks for deterministic constraints and `verify(...)` conditions for semantic quality constraints.
- Include positive and negative examples that are strong enough for one-shot validation.
- Reuse the `TaskCreated` structure directly from the interface described in this skill.
- Name helper files by content so the directory reads clearly.

## Expected Output

After the runner succeeds, the directory should usually contain:

- one content-reflecting runner file such as `define_concept_described.py`
- one generated task file with a content-reflecting name such as `concept_described.py`
- any example or support files needed by the verification examples

The generated task file should already include embedded examples produced by `TaskCreated`.

## Minimal Example

For a request like `/hdr-define describe a concept in markdown`, prefer a structure like:

```python
from hdr.tasks.meta import FieldSpec, TaskCreated, VerifySpec
from hdr.tasks.coding import MarkdownFileWritten

context_file = MarkdownFileWritten(path="framework_context.md")
good_description = MarkdownFileWritten(path="concept_description.md")
bad_description = MarkdownFileWritten(path="concept_description_too_vague.md")

concept_described_task = TaskCreated(
    class_name="ConceptDescribed",
    docstring="Represent a concept described within a known context.",
    imports=[
        "from pydantic import Field",
        "from hdr.tasks.std import Task",
        "from hdr.tasks.coding import MarkdownFileWritten",
    ],
    fields=[
        FieldSpec(name="context", type_annotation="MarkdownFileWritten", description="Context file the reader already knows"),
        FieldSpec(name="name", type_annotation="str", description="Concept name"),
        FieldSpec(name="description", type_annotation="MarkdownFileWritten", description="Markdown file describing the concept"),
    ],
    verifies=[
        VerifySpec(
            condition="The description lets a reader who knows context understand name clearly and precisely.",
            positive_example={"context": context_file, "name": "HDR Task", "description": good_description},
            negative_example={"context": context_file, "name": "HDR Task", "description": bad_description},
        )
    ],
)
```

This will generate `concept_described.py`.

Save this in a runner file such as `define_concept_described.py`.

## Execution

Run the content-reflecting runner file with naked Python:

```bash
python define_concept_described.py
```

Work in the Python environment where `hdr` is importable.
