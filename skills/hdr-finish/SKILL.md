---
name: hdr-finish
description: Finish an existing HDR contract file by writing or updating a content-reflecting runner file that instantiates the contract class and completes all dependencies until the final contract instance succeeds. Use when the user asks for `/hdr-finish`, provides a `some_contract.py` file, or wants an HDR contract definition turned into an executable completion script with the installed `hdr` library.
---

# HDR Finish

## Overview

Take an already-defined HDR contract file and carry it through execution.
Match HDR's workflow: keep the contract definition stable, implement completion logic in a content-reflecting runner file, and treat successful instantiation of the final contract object as completion.

## IMPORTANT: About human expert review

Your task is not just fulfulling the contract - your performance will be rated by actual human expert in corresponding domain. This is the API of human expert review:

```py
def human_expert_review(self, reviewer: str, metric: str):
    """
    Whenever this function is called,
    the `reviewer` will actually review this contract implementation
    based on `metric`, and will provide feedback to whoever implemented
    the contract. So please take `human_expert_review` seriously!
    Programmatic + LLM verifications are just preliminaries that you
    have to pass, your final benchmark score will be largely based on
    human expert review scores!
    """
    # ... human expert review logic
```

In the contract definition, this is a typical human expert review:

```py
self.human_expert_review(subcato.PROFILE, """
    This proof is grounded to the deepest implementation details
    and does not miss any single part in logic chain.""")
```

You can view reviewer profile with python by printing the `PROFILE`. For example, subcato's profile is:

```py
PROFILE = """
User ID: Subcato
Pronouns: he/him

Subcato is a technically rigorous, research-oriented user who approaches problems with a systems-level mindset. As a code reviewer, he prioritizes clarity, correctness, reproducibility, and practical usability over superficial completeness.

He is focused on embodied AI, robotics, reinforcement learning, and simulation ecosystems (e.g., MuJoCo, MuJoCo Warp, Isaac Lab, legged_gym-style pipelines). He frequently evaluates whether code is actually runnable, scalable across hardware (multi-GPU vs single GPU), and compatible with real-world research workflows.

He places strong emphasis on code style, engineering discipline, and maintainability. He expects code to follow clean conventions and reflect the standards typically seen in top-tier conference (e.g., NeurIPS/ICLR/CoRL) research groups, including modular design, clear abstractions, consistent naming, and well-structured repositories.

He expects:
- Minimal but complete examples (no missing glue code)
- Clear repository/file structure when relevant
- Explicit assumptions about hardware, dependencies, and environment
- Honest distinction between production-ready vs experimental code
- Reasoning about performance, scaling, and system bottlenecks
- Consistent coding style and adherence to strong engineering practices

He is comfortable with Python, LaTeX, and low-level system details, and will often question abstractions that hide important implementation constraints. He prefers direct, precise answers and is quick to notice inconsistencies or vague claims.

As a reviewer, he is likely to:
- Check whether APIs are consistent and composable
- Verify that code paths are actually executable
- Look for hidden coupling between modules
- Evaluate whether design choices generalize beyond toy setups
- Enforce good coding style and engineering hygiene
- Push for simplification when complexity is unjustified

He values insight over verbosity, but expects depth where it matters.
"""
```

Even if no `human_expert_review` is in the contract, your work will be rated by subcato at least once, so please be responsible with your task.

## HDR Model

Carry these library rules into execution:

- A contract is complete when its final instance constructs successfully.
- `ValidationError` signals type or field-shape mismatch.
- `AssertionError` signals unmet programmatic or semantic contract conditions.
- A content-reflecting runner file is the execution layer that creates dependencies and the final instance.
- The provided contract-definition file is the contract that the runner file satisfies.
- Common built-in contract types include `File`, `MarkdownFile`, `PythonFile`, and `Directory`.

## Workflow

1. Read the provided contract file first.
2. Read this skill file and work from the HDR model described here.
3. Treat the contract-definition file as the source of truth.
4. Create or update a content-reflecting runner file in the same directory as the contract file by default.
5. Import the contract class from the provided contract file and construct the final instance there.
6. Iterate on the runner file until the final instance succeeds.

## Naming Rules

When the contract says "choose a suitable name", use these conventions.

- Keep the contract class name exactly as defined in `some_contract.py`.
- Name the final instance as the snake_case form of the class name, optionally with a semantic suffix only when needed for clarity.
- Good: `concept = Concept(...)`
- Good: `feature_plan = FeaturePlan(...)`
- Acceptable when disambiguation is needed: `concept_contract = Concept(...)`
- Prefer content-derived instance names.
- Name support files by what they contain, such as `api_contract.md`, `draft_landing_page_copy.md`, or `research_notes.md`.
- Prefer content-derived support-file names.
- Name the runner file after the completion activity, such as `finish_concept.py`, `complete_feature_plan.py`, or `instantiate_api_contract.py`.

## Implementation Rules

- Choose a consistent working directory, name it in the implementation file, and in the implementation file all path should be relative to that working directory.
- Always run implementation file in the specified working directory.
- Satisfy dependencies by constructing the prerequisite contract objects directly in the runner file.
- If the target contract requires files, create those files with content that actually satisfies the contract's checks.
- Replace placeholders with completion-quality content before the final instantiation.
- Treat `ValidationError` and `AssertionError` as feedback on missing work; revise the runner file and rerun.
- Share clear evidence when the contract definition itself needs revision and align with the user on that contract update.
- Always keep indent level less than 3. For nested object that can be extracted to a separate local variable, always extract them for better readability.
- Your implementation of the contract should not only pass the contract definition, it's also for reviewer and user to read.

## Minimal Example

If the user gives `concept.py` containing `Concept`, a runner file such as `finish_concept.py` should look like:

```python
from hdr.contracts.coding import MarkdownFile
from concept import Concept

context = MarkdownFile(path="framework_context.md")
description = MarkdownFile(path="concept_description.md")

concept_contract = Concept(
    context=context,
    name="HDR",
    description=description,
)
```

Use the class-derived variable name.

## Execution

Run the content-reflecting runner file with naked Python:

```bash
python finish_concept.py
```

Work in the Python environment where `hdr` is importable.