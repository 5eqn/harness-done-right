---
name: hdr-finish
description: Finish an existing HDR task file by writing or updating a content-reflecting runner file that instantiates the task class and completes all dependencies until the final task instance succeeds. Use when the user asks for `/hdr-finish`, provides a `some_task.py` file, or wants an HDR task definition turned into an executable completion script with the installed `hdr` library.
---

# HDR Finish

## Overview

Take an already-defined HDR task file and carry it through execution.
Match HDR's workflow: keep the task definition stable, implement completion logic in a content-reflecting runner file, and treat successful instantiation of the final task object as completion.

## HDR Model

Carry these library rules into execution:

- A task is complete when its final instance constructs successfully.
- `ValidationError` signals type or field-shape mismatch.
- `AssertionError` signals unmet programmatic or semantic task conditions.
- A content-reflecting runner file is the execution layer that creates dependencies and the final instance.
- The provided task-definition file is the contract that the runner file satisfies.
- Common built-in task types include `FileWritten`, `MarkdownFileWritten`, `PythonFileWritten`, and `DirectoryCreated`.

## Workflow

1. Read the provided task file first.
2. Read this skill file and work from the HDR model described here.
3. Treat the task-definition file as the source of truth.
4. Create or update a content-reflecting runner file in the same directory as the task file by default.
5. Import the task class from the provided task file and construct the final instance there.
6. Iterate on the runner file until the final instance succeeds.
7. Ask the user to run the agent inside the Python environment where `hdr` is installed.

## Naming Rules

When the task says "choose a suitable name", use these conventions.

- Keep the task class name exactly as defined in `some_task.py`.
- Name the final instance as the snake_case form of the class name, optionally with a semantic suffix only when needed for clarity.
- Good: `concept_described = ConceptDescribed(...)`
- Good: `feature_plan_drafted = FeaturePlanDrafted(...)`
- Acceptable when disambiguation is needed: `concept_described_task = ConceptDescribed(...)`
- Prefer content-derived instance names.
- Name support files by what they contain, such as `api_contract.md`, `draft_landing_page_copy.md`, or `research_notes.md`.
- Prefer content-derived support-file names.
- Name the runner file after the completion activity, such as `finish_concept_described.py`, `complete_feature_plan_drafted.py`, or `instantiate_api_contract_written.py`.

## Implementation Rules

- Prefer relative paths and local artifacts.
- Satisfy dependencies by constructing the prerequisite task objects directly in the runner file.
- If the target task requires files, create those files with content that actually satisfies the task's checks.
- Replace placeholders with completion-quality content before the final instantiation.
- Treat `ValidationError` and `AssertionError` as feedback on missing work; revise the runner file and rerun.
- Share clear evidence when the task definition itself needs revision and align with the user on that contract update.

## Minimal Example

If the user gives `concept_described.py` containing `ConceptDescribed`, a runner file such as `finish_concept_described.py` should look like:

```python
from hdr.tasks.coding import MarkdownFileWritten
from concept_described import ConceptDescribed

context = MarkdownFileWritten(path="framework_context.md")
description = MarkdownFileWritten(path="concept_description.md")

concept_described = ConceptDescribed(
    context=context,
    name="HDR Task",
    description=description,
)
```

Use the class-derived variable name.

## Execution

Run the content-reflecting runner file with naked Python:

```bash
python finish_concept_described.py
```

Work in the Python environment where `hdr` is importable.
