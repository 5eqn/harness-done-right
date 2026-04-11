---
title: Finish HDR Tasks With Content-Reflecting Runners
date: 2026-04-11
category: docs/solutions/workflow-issues/
module: HDR task execution
problem_type: workflow_issue
component: development_workflow
severity: medium
applies_when:
  - Completing an existing generated HDR task file
  - Turning an HDR task contract into executable completion logic
  - Building a sequence of atomic task instances with accumulated context
  - Verifying HDR examples that call semantic self.verify checks
tags: [hdr, hdr-finish, movechosen, runners, verification, workflow]
---

# Finish HDR Tasks With Content-Reflecting Runners

## Context

Finishing `examples/build_move_task/move_chosen.py` required a different move
from defining it. The generated `MoveChosen` task was already the contract: it
held fields for `context`, `purpose`, `content`, `contrast`, `before`, and
`after`, plus programmatic and semantic checks. The right completion step was
not to edit that generated file, but to create a runner that satisfies it.

The completed runner, `examples/build_move_task/complete_move_chosen.py`, builds
a seven-step chain of atomic `MoveChosen` instances that introduces HDR. Each
accepted move is converted into a `MoveSnapshot` and carried forward in
`MoveContext.prior_moves`, so the next move is validated against the accumulated
reader state.

Session history showed related prior learning about defining `MoveChosen` with
small fields and demanding verification, but not this specific finish-runner
workflow (session history). That makes this note complementary to
`docs/solutions/best-practices/hdr-task-definition-workflow-2026-04-11.md`
rather than a replacement.

## Guidance

Treat the task-definition file as the stable contract. For generated files such
as `move_chosen.py`, finish the task by writing a content-reflecting runner next
to it:

```python
from hdr.tasks.coding import MarkdownFileWritten

from move_chosen import MoveChosen
from move_support import MoveContext, MovePurpose, MoveSnapshot


target_reader = MarkdownFileWritten(path="hdr_target_reader.md")
target_meaning = MarkdownFileWritten(path="hdr_target_meaning.md")
prior_moves: list[MoveSnapshot] = []
```

Use the runner to construct dependencies directly. When the task represents one
atomic move in a larger chain, keep the task atomic and let the runner manage
the chain:

```python
def move_context() -> MoveContext:
    return MoveContext(
        target_reader=target_reader,
        target_concept="HDR",
        target_meaning=target_meaning,
        prior_moves=[*prior_moves],
    )


def choose_move(...) -> MoveChosen:
    move_chosen = MoveChosen(context=move_context(), ...)
    prior_moves.append(
        MoveSnapshot(
            purpose=move_chosen.purpose,
            content=move_chosen.content,
            contrast=move_chosen.contrast,
            before=move_chosen.before,
            after=move_chosen.after,
        )
    )
    return move_chosen
```

For `MoveChosen`, the completed chain used seven single-axis moves:

- `named vs inspected`
- `suggestive vs contractual`
- `declared vs constructed`
- `implicit vs typed`
- `malformed vs well-shaped`
- `performative vs semantic`
- `plausible vs complete`

Run the runner from the directory that owns its local artifacts, using relative
paths directly:

```bash
cd examples/build_move_task
uv run python complete_move_chosen.py
```

Do not set `PYTEST_CURRENT_TEST` for the final proof run. That environment path
is useful for local shape checks because it mocks `Task.verify()`, but final HDR
completion should exercise the real verifier and finish only when the final
task instance constructs successfully.

## Why This Matters

HDR completion is construction, not prose summary. A runner that instantiates
the task object is the artifact that proves the work is done. For a chain task,
the runner should make the sequence explicit without changing the atomic task
contract.

The `MoveSnapshot` handoff matters because it avoids recursive `MoveChosen`
objects while still preserving reader-state continuity. Each move remains one
validated transition, and the next move sees only the stable summary of prior
accepted moves.

Visible verifier output also matters for this workflow. The first direct run
appeared stuck because successful `self.verify()` logs were buffered. Adding
`flush=True` to the verification success print made long real-verifier runs
observable:

```python
print(f"[verify] score={score} {_summarize_condition(condition)}", flush=True)
```

The successful rerun emitted score-5 verification logs and ended with:

```text
MoveChosen chain completed with 7 atomic moves.
```

## When to Apply

- When `/hdr-finish` is applied to an existing task-definition file
- When the task file is generated and should remain stable
- When completion requires constructing several prerequisite or atomic task
  instances before the final task instance can succeed
- When semantic `self.verify()` calls make final verification long enough that
  progress logs should flush immediately

## Examples

Less effective finish workflow:

```text
Edit the generated MoveChosen class until it contains the whole HDR
introduction.
```

Better finish workflow:

```text
Leave MoveChosen as the atomic contract. Write complete_move_chosen.py to build
each accepted move, append a MoveSnapshot, and let the final MoveChosen
construction prove completion.
```

Less effective verification:

```bash
PYTEST_CURRENT_TEST=hdr_finish uv run python examples/build_move_task/complete_move_chosen.py
```

That checks shape with mocked verification. Use it only as a debugging shortcut.
For final completion, run the real verifier from the example directory:

```bash
cd examples/build_move_task
uv run python complete_move_chosen.py
```

## Related

- `docs/solutions/best-practices/hdr-task-definition-workflow-2026-04-11.md`
- `docs/solutions/best-practices/hdr-contract-evolution-directions-2026-04-11.md`
- `docs/solutions/documentation-gaps/hdr-agent-harness-definition-2026-04-11.md`
- `examples/build_move_task/complete_move_chosen.py`
- `examples/build_move_task/move_chosen.py`
- `examples/build_move_task/move_support.py`
- `src/hdr/tasks/std.py`
