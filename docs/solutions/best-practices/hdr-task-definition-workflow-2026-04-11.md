---
title: Define HDR Tasks as Small Fields With Demanding Verifications
date: 2026-04-11
category: docs/solutions/best-practices/
module: HDR task authoring
problem_type: best_practice
component: documentation
severity: medium
applies_when:
  - Planning a new HDR task from a novel use case
  - Using TaskCreated to generate a task definition
  - Turning subjective quality or taste into an executable contract
  - Designing examples under examples/
tags: [hdr, taskcreated, task-design, verification, examples, taste]
---

# Define HDR Tasks as Small Fields With Demanding Verifications

## Context

The `MoveChosen` planning session started from a failed shape: broad
storytelling validations over a completed concept explanation. Those checks
were too easy for a fluent model to satisfy, so they did not create a useful
critique loop.

The better design was to make the task constructive. Instead of validating a
finished paragraph after the fact, the task represents one chosen conceptual
move before final prose exists. The final text is compiled from those moves.

Session history search found related earlier HDR work around `TaskCreated`,
`self.verify()`, and introduction-writing examples, but the concrete
`MoveChosen` workflow came from the current session (session history). The
older context reinforces the same lesson: a broad `self.verify()` can validate
an artifact, but a good HDR task makes the hidden contract explicit.

## Guidance

Keep the task fields few and load the rigor into orthogonal verifications.

The final `MoveChosen` task surface is intentionally small:

```python
class MoveChosen(Task):
    context: MoveContext
    purpose: MovePurpose
    content: str
    contrast: str
    before: str
    after: str
```

Each field must earn its place:

- `context` carries target reader, target concept, target meaning, and prior
  moves so necessity can be checked against the whole intended expression.
- `purpose` selects the validation family. In the first experiment,
  `BUILD` creates concrete pressure and `SHIFT` changes the reader's lens.
- `content` is the atomic outline sentence, not final prose.
- `contrast` is the projected one-dimensional axis of the move, such as
  `soft vs enforceable`.
- `before` and `after` describe the focused local reader lens changed by the
  move.

Use support models when `TaskCreated` should not generate every type. The
`MoveChosen` experiment uses `MoveSnapshot` rather than a recursive
`MoveChosen` reference because `TaskCreated` handles one generated class
cleanly, while recursive generated examples add avoidable friction:

```python
class MoveContext(BaseModel):
    target_reader: MarkdownFileWritten
    target_concept: str
    target_meaning: MarkdownFileWritten
    prior_moves: list[MoveSnapshot]
```

Do not make every quality criterion a field. For taste-heavy tasks, many
criteria are judgments about whether fields faithfully represent each other.
Those belong in `VerifySpec` conditions with strong positive and negative
examples.

The useful split is:

- **General checks**: contrast axis is valid, fields share one axis, content
  causes `after`, content is necessary against `target_meaning`, and the move
  uses only concepts introduced by the target reader or prior moves.
- **Purpose-specific checks**: `BUILD` must add concrete pressure, load-bearing
  detail, and preparation without premature resolution. `SHIFT` must produce
  an earned lens change, irreversible cognition upgrade, and no scene reset.

Make every verification atomic. Each condition should test one reason the task
could be invalid. Avoid broad checks like "the sentence is good" or mixed
checks that combine necessity, vividness, and reader fit in one assertion.

Make every verification demanding. The negative examples should fail for
middle-standard output, not only obviously broken output. In this session, a
negative example that said "prose is cheaper, task objects are more enforceable
and easier to test" was too close to the intended `soft vs enforceable` axis,
so the LLM verifier scored it as passing. Replacing it with a cleaner setup-cost
example made the contract sharper.

## Why This Matters

HDR is strongest when the task object becomes the working contract, not a
post-hoc rubric. Small fields make the task legible. Strong verifications make
the task bite.

For subjective or taste-like work, the temptation is to add many flat fields:
`why_better`, `reader_delta`, `purpose`, `alternatives`, `mood`, and so on. That
can make the model fill out a form rather than think. A smaller contract with
cross-field checks is often better: each field represents the artifact's
backbone, and the verifier checks whether those fields cohere.

The `target_meaning` field is important. Without the whole conceptual target,
the verifier can pretend any vivid move is necessary. With `target_meaning`, it
can judge whether a move is hard to delete from an excellent introduction.

## When to Apply

- When a new HDR example is novel enough that there is no obvious existing task
  shape
- When the task is trying to encode quality, taste, judgment, or conceptual
  structure
- When a generated artifact should be constructed from intermediate decisions
  rather than inspected after completion
- When `TaskCreated` needs helper types that are clearer as hand-written
  support models

## Examples

Less effective task planning:

```text
Make a ConceptDescribed task with seven global storytelling checks over the
final markdown file.
```

Better task planning:

```text
Represent the outline move that final prose will later express. Keep fields
minimal: context, purpose, content, contrast, before, after. Make the verifier
prove that those fields describe one necessary cognition shift along one valid
contrast axis.
```

Less effective contrast:

```text
high cost vs low guarantee
```

That bundles two dimensions. Use a projected opposition instead:

```text
soft vs enforceable
```

Concrete generated example:

```python
move_chosen_task = TaskCreated(
    class_name="MoveChosen",
    fields=[
        FieldSpec(name="context", type_annotation="MoveContext", ...),
        FieldSpec(name="purpose", type_annotation="MovePurpose", ...),
        FieldSpec(name="content", type_annotation="str", ...),
        FieldSpec(name="contrast", type_annotation="str", ...),
        FieldSpec(name="before", type_annotation="str", ...),
        FieldSpec(name="after", type_annotation="str", ...),
    ],
    verifies=[
        VerifySpec(
            condition="The contrast is exactly one projected axis...",
            positive_example={...},
            negative_example={...},
        ),
    ],
)
```

Verified commands from the `MoveChosen` definition:

```bash
uv run python define_move_chosen.py
uv run ruff check examples/build_move_task
uv run pyright examples/build_move_task
uv run pytest
```

## Related

- `docs/solutions/workflow-issues/hdr-finish-content-reflecting-runners-2026-04-11.md`
- `docs/solutions/best-practices/hdr-contract-evolution-directions-2026-04-11.md`
- `docs/solutions/documentation-gaps/hdr-agent-harness-definition-2026-04-11.md`
- `examples/build_move_task/define_move_chosen.py`
- `examples/build_move_task/move_support.py`
- `examples/build_move_task/move_chosen.py`
- `src/hdr/tasks/meta.py`
