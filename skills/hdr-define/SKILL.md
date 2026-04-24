---
name: hdr-define
description: Define a new HDR contract from a plain-language contract description by writing a content-reflecting runner file that instantiates `hdr.contracts.meta.Contract` and generates a content-reflecting contract file. Use when the user asks for `/hdr-define`, wants to formalize a new contract with the installed `hdr` library, or wants a contract type created from requirements instead of writing the class by hand.
---

# HDR Define

## Overview

Turn a natural-language request into a proper HDR contract definition.
Follow HDR's core philosophy: contracts are values, validation happens at construction time, and the generated contract file should be specific, typed, and self-validating.

## HDR Model

Carry these library rules into every contract:

- A completed contract is an instance of a Python class.
- `BaseContract` extends Pydantic, so field types validate at instantiation.
- `self.llm_verify(condition)` evaluates semantic conditions against the full contract state.
- Verification results are cached by condition, so reruns are cheap and stateless.
- `Contract` creates a concrete contract class, writes the generated Python file, validates the file with `PythonFile`, and embeds positive and negative examples into the generated `llm_verify(...)` strings.

## Contract Interface

Write against this interface directly:

```python
Contract(
    class_name: str,
    parent_class: str = "BaseContract",
    docstring: str,
    imports: list[str] = [
        "from pydantic import Field",
        "from hdr.contracts.std import BaseContract",
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

`Contract` derives the generated file name from `class_name` by converting `PascalCase` to `snake_case.py`.

## Workflow

1. Read this skill file and work from the interface described here.
2. Use `.` as the working directory by default.
3. Create a content-reflecting runner file in that directory.
4. Build one `Contract(...)` instance inside that runner file.
5. Let `Contract` generate a specific snake_case contract-definition file derived from the class name.
6. Run `python <content_reflecting_runner>.py` from that directory.
7. Ask the user to run the agent inside the Python environment where `hdr` is installed.

## Naming Rules

- Choose a contract class name in `PascalCase` that states the completed artifact or state.
- Prefer a narrow noun phrase, such as `Concept`, `ApiContract`, `FeaturePlan`, or `LandingPageCopy`.
- Prefer content-reflecting names over library-reflecting names.
- Since `Contract` derives the file name from `class_name`, choose the class name so the snake_case file name is also specific and readable.
- Use a generated file name that reflects content, such as `concept.py` or `feature_plan.py`.
- Name the runner file after the contract-definition activity, such as `define_concept.py`, `build_feature_plan.py`, or `create_api_contract.py`.

## Authoring Rules

- Build the contract by creating one `Contract(...)` instance in the content-reflecting runner file.
- Use `FieldSpec` and `VerifySpec` explicitly instead of raw unstructured dicts when practical.
- Give every field a concrete description.
- Prefer relative paths.
- Keep programmatic checks for deterministic constraints and `llm_verify(...)` conditions for semantic quality constraints.
- Include positive and negative examples that are strong enough for one-shot validation.
- Reuse the `Contract` structure directly from the interface described in this skill.
- Name helper files by content so the directory reads clearly.

## Expected Output

After the runner succeeds, the directory should usually contain:

- one content-reflecting runner file such as `define_concept.py`
- one generated contract file with a content-reflecting name such as `concept.py`
- any example or support files needed by the verification examples

The generated contract file should already include embedded examples produced by `Contract`.

## Minimal Example

For a request like `/hdr-define describe a concept in markdown`, prefer a structure like:

```python
from hdr.contracts.meta import FieldSpec, Contract, VerifySpec
from hdr.contracts.coding import MarkdownFile

context_file = MarkdownFile(path="framework_context.md")
good_description = MarkdownFile(path="concept_description.md")
bad_description = MarkdownFile(path="concept_description_too_vague.md")

concept_contract = Contract(
    class_name="Concept",
    docstring="Represent a concept defined within a known context.",
    imports=[
        "from pydantic import Field",
        "from hdr.contracts.std import BaseContract",
        "from hdr.contracts.coding import MarkdownFile",
    ],
    fields=[
        FieldSpec(name="context", type_annotation="MarkdownFile", description="Context file the reader already knows"),
        FieldSpec(name="name", type_annotation="str", description="Concept name"),
        FieldSpec(name="description", type_annotation="MarkdownFile", description="Markdown file defining the concept"),
    ],
    verifies=[
        VerifySpec(
            condition="The description lets a reader who knows context understand name clearly and precisely.",
            positive_example={"context": context_file, "name": "HDR Contract", "description": good_description},
            negative_example={"context": context_file, "name": "HDR Contract", "description": bad_description},
        )
    ],
)
```

This will generate `concept.py`.

Save this in a runner file such as `define_concept.py`.

## Execution

Run the content-reflecting runner file with naked Python:

```bash
python define_concept.py
```

Work in the Python environment where `hdr` is importable.
