---
title: Treat HDR as Contract Evolution, Not Just LLM Validation
date: 2026-04-11
category: docs/solutions/best-practices/
module: HDR product direction
problem_type: best_practice
component: documentation
severity: medium
applies_when:
  - Framing HDR use cases beyond simple semantic validation
  - Designing new HDR examples for research, analysis, or self-improving tasks
  - Evolving TaskCreated or meta-task capabilities
  - Explaining why concept-writing demos are stress tests rather than the product
tags: [hdr, contract-evolution, taskcreated, research, analysis, taste]
---

# Treat HDR as Contract Evolution, Not Just LLM Validation

## Context

The concept-writing and `MoveChosen` experiments clarified an important product
direction: the demo is a small, subjective stress test, not necessarily the
primary use case. Global storytelling validations can pass too easily because a
fluent LLM draft can satisfy broad rubrics without forming a useful critique
loop.

The stronger framing is that HDR should help agents construct and evolve task
contracts. A task contract is not just a post-hoc rubric over an artifact. It is
the structure the agent must satisfy while producing the artifact. This matters
because a normal prompt, checklist, or broad `self.verify()` assertion can still
leave "done" as a conversational claim.

Session history also showed earlier HDR introductions being scored as
incomplete when they framed HDR mostly as a Python library for task validation.
Those attempts were useful, but they risked collapsing the project into "AI
assertions with Pydantic" instead of preserving the deeper harness claim
(session history).

All directions below are provisional. They are current working hypotheses from
conversation, waiting for more examples, tests, and discussion.

## Guidance

Frame HDR as a contract-evolution framework for agents:

```text
define contract -> instantiate task -> observe output quality -> compare contracts -> revise contract
```

This is broader than validating a final answer. The contract itself becomes an
artifact the agent can improve.

Promising use-case directions:

- **Deep research**: enforce reference title-link matching, source existence,
  concept extraction, and concept evolution across sources. A `TitleLinkPaired`
  task is a normal task relation, not a decision primitive.
- **Novel analysis**: enforce reproducible reverse engineering of how a text was
  made. The output should not be only an interpretation; it should be a
  contract someone else can replay or instantiate.
- **Self-evolution**: given a core value for a task, compare two task contracts
  by generating instances from each and evaluating output quality. Use the
  comparison as feedback for improving the task design itself.
- **Meta-task improvement**: `TaskCreated` is part of the product surface. Its
  limitations, such as poor fit for recursive helper types and the need for
  strong positive/negative examples, should inform future meta-task design.

Keep the concept/move demo in its proper place. It is useful because subjective
taste stresses weak contracts quickly. It should not be treated as proof that
storytelling is the main use case.

## Why This Matters

If HDR is explained as "LLM validation for task objects," it sounds like a small
library feature. If HDR is explained as "the harness makes completion depend on
constructing an explicit contract," the idea becomes a different programming
model for agent work.

The distinction changes what to build next. A validation-library roadmap would
add more predicates and examples. A contract-evolution roadmap asks how agents
can generate, compare, and improve task definitions themselves.

This also explains why broad global rubrics are not enough. A broad rubric can
approve competent output. A contract should make the hidden structure of the
work explicit enough that an agent receives specific failure pressure.

## When to Apply

- When deciding which HDR example should become the next public demo
- When reviewing whether a new task is a hard contract or a soft checklist
- When designing research or analysis examples that need reproducible evidence
- When considering changes to `TaskCreated`, `VerifySpec`, generated helper
  types, or task comparison workflows

## Examples

Less precise framing:

```text
HDR validates that an LLM output satisfies a task.
```

Better framing:

```text
HDR makes the agent satisfy an explicit task contract, and the contract itself
can become something the agent improves.
```

For deep research, avoid treating every useful relation as a "decision." A title
and source link matching is a completed relation task:

```text
TitleLinkPaired
```

A decision-like primitive should be reserved for cases where the agent chooses
among meaningful alternatives. The `MoveChosen` prototype uses this distinction
for taste work, but it remains a subjective stress test:

```text
MoveContext + MovePurpose + content + contrast + before + after
```

The general lesson is not that all writing should use `MoveChosen`. The lesson
is that contracts get stronger when they expose the hidden choices or evidence
relations the agent would otherwise leave implicit.

## Related

- `docs/solutions/workflow-issues/hdr-finish-content-reflecting-runners-2026-04-11.md`
- `docs/solutions/documentation-gaps/hdr-agent-harness-definition-2026-04-11.md`
- `examples/build_move_task/define_move_chosen.py`
- `examples/build_move_task/move_support.py`
- `examples/build_concept_task/define_concept_described.py`
- `src/hdr/tasks/meta.py`
