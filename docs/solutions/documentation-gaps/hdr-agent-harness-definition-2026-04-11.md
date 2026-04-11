---
title: Explain HDR Using the Agent Harness Meaning
date: 2026-04-11
category: docs/solutions/documentation-gaps/
module: HDR concept documentation
problem_type: documentation_gap
component: documentation
severity: medium
applies_when:
  - Writing introductory or concept-level explanations of HDR
  - Describing Harness Done Right to readers who already know AI agents
  - Creating examples under examples/build_concept_task/
tags: [hdr, harness, agents, concept-writing, documentation]
---

# Explain HDR Using the Agent Harness Meaning

## Context

While finishing `examples/build_concept_task/concept_described.py`, the first
HDR concept description treated "harness" as if it meant a test harness that
runs work and reports pass or fail. The user corrected the framing: in this
project, "harness" means the framework around an AI model that handles context
window, memory, tool use, and control flow. The underlying definition is
`Agent = Model + Harness`.

Prior session history also showed older HDR wording centered on task
formalization and successful instantiation as proof of completion, but not this
specific agent-harness distinction (session history). That makes the
distinction worth documenting because future HDR explanations can otherwise
drift toward a generic testing interpretation.

## Guidance

When explaining HDR, assume the reader may know AI agents but may not know this
project's specific HDR idea. Ground the explanation in the agent runtime
definition:

```text
Agent = Model + Harness
```

Use "harness" to mean the agent framework layer around the model: context
management, memory, tool use, and control flow. Then describe HDR as the move of
making that harness guide the agent through explicit task objects that must
successfully instantiate.

The completed concept example follows this shape:

```python
concept_described = ConceptDescribed(
    target_reader=target_reader,
    name="HDR",
    description=description,
)
```

The supporting target-reader file should include the agent-harness background:

```markdown
- The definition `Agent = Model + Harness`
- Harnesses as the runtime framework around a model, including context window,
  memory, tool use, and control flow
```

Avoid describing HDR as a testing harness. HDR can use validation and
verification, but the core idea is not "tests around output." It is a harness
design pattern for agent work: formalize the intended outcome as a task object,
let typed fields and semantic `self.verify()` checks define the contract, and
treat successful construction of the final instance as completion.

## Why This Matters

The word "harness" is overloaded. If it is read as "test harness," HDR sounds
like an after-the-fact checking mechanism. That loses the important design
claim: the harness is part of the agent itself, and HDR changes how that harness
guides the agent's work before the final answer exists.

The corrected explanation also makes the boundary of HDR clearer. A normal
prompt, a checklist in memory, or a post-hoc review is not HDR by itself. An
agent system is using HDR when the harness routes work through explicit task
contracts and completion means the final task instance constructs successfully.

## When to Apply

- When writing README, skill, or onboarding copy for HDR
- When building `ConceptDescribed` examples for HDR or related agent concepts
- When a draft uses "harness" in a way that could be mistaken for testing
  infrastructure
- When explaining why HDR is about agent execution structure, not just
  validation

## Examples

Less precise:

```markdown
HDR is a test harness that runs agent work and reports whether it passes.
```

Better:

```markdown
HDR is a way to make the agent harness guide work through explicit task
objects. In Harness Done Right, the contract lives as a Python task class:
fields name the required evidence, Pydantic checks the shape, and
`self.verify()` assertions judge semantic conditions against the constructed
object.
```

In the verified example, the runner was:

```python
from hdr.tasks.coding import MarkdownFileWritten

from concept_described import ConceptDescribed


target_reader = MarkdownFileWritten(path="hdr_target_reader.md")
description = MarkdownFileWritten(path="hdr_description.md")

concept_described = ConceptDescribed(
    target_reader=target_reader,
    name="HDR",
    description=description,
)
```

This passed all seven `ConceptDescribed` semantic checks at score 5 and printed:

```text
ConceptDescribed task completed: HDR
```

## Related

- `docs/solutions/workflow-issues/hdr-finish-content-reflecting-runners-2026-04-11.md`
- `examples/build_concept_task/hdr_target_reader.md`
- `examples/build_concept_task/hdr_description.md`
- `examples/build_concept_task/finish_concept_described.py`
