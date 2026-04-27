# HDR (Harness Done Right) — Philosophy & Innovation Catalog

## What is HDR?

HDR is a **contract-as-code framework** where requirements are executable
Python classes that enforce themselves at construction time. Instead of
writing specs in prose and hoping the implementation matches, you define a
contract and the framework verifies it — using both programmatic checks and
LLM semantic judgment.

---

## Philosophy

### 1. Contracts are executables, not documents

A specification that can't verify itself is just a suggestion. HDR contracts
are Python classes. When you instantiate one, all validations fire
immediately in `__init__`. There is no separate "check" step — construction
is verification. If a contract fails, you get an exception, not a report to
ignore.

### 2. Two-layer verification: deterministic + semantic

Programmatic checks (types, file existence, linters, type-checkers) are
fast, cheap, and deterministic. But they can't judge quality, clarity, or
semantic correctness. HDR layers LLM-based verification on top for the
qualitative properties that matter most. You get both: the speed of
traditional validation for the mechanical stuff, and the judgment of an LLM
for the human stuff.

### 3. Agents should prove their own output

HDR is designed for the agentic era. An AI agent working on a task should
define what "done" looks like as a contract, then instantiate that contract
to verify its own output. The contract becomes the interface between the
agent and the codebase — the agent doesn't just produce output, it proves
the output meets the spec.

### 4. Examples are the universal specification language

Every LLM verify condition in HDR embeds a positive example (score=5) and a
negative example (score=1). This anchors the LLM's judgment in concrete
cases, eliminating ambiguity. When you say "the code is well-documented,"
the examples show exactly what you mean.

### 5. Contracts compose and inherit

`PythonFile` is a `File` with additional constraints. `PythonWorkspace` is a
`Directory` that also runs pyright and ruff. `Concept` contains
`MarkdownFile` fields and adds semantic checks. The contract hierarchy
mirrors real-world domain relationships, and validation composes
automatically.

---

## What's Innovative

### 1. `llm_verify()` as a language primitive

`self.llm_verify(condition)` is a method on the base contract class that
calls an LLM to judge a semantic condition on a 1-5 scale. It's invoked in
`__init__`, just like a Pydantic validator. The LLM sees the full
pretty-printed contract state, so conditions can reference any field by name
without manual f-string wiring.

This makes LLM judgment a **first-class verification primitive** alongside
type checks and assertions — not an afterthought bolted onto a separate
evaluation pipeline.

### 2. Meta-contract: contracts that write contracts

`Contract` is a meta-contract. You give it a class name, field specs, and a
list of `VerifySpec` (each with a condition, positive example, and negative
example). It:

- Validates the specification itself (are all required fields present? do
  examples cover them?)
- **LLM-verifies each example** against its condition before accepting the
  spec
- **Auto-generates** a complete `.py` file with the contract class, passing
  PyRight and Ruff checks
- **Refuses to overwrite** any file not marked as
  `# HDR-AUTO-GENERATED`

The meta-contract is a contract whose successful execution *produces another
contract*. This is contracts all the way down.

### 3. Embedded examples in generated code

When the meta-contract generates a contract class, each verify call includes
the positive and negative examples inline:

```python
self.llm_verify(
    "The description is written for readers who "
    "understand context but do not yet know name...\n\n"
    "Example that PASSES (score 5):\n"
    "Concept(\n  context = MarkdownFile(\n"
    "    path = 'context.md',\n    ...\n"
    "Example that FAILS (score 1):\n"
    "Concept(\n  context = MarkdownFile(\n"
    "    path = 'context.md',\n    ...\n"
)
```

This means generated contracts are self-documenting and self-calibrating —
anyone reading the code sees exactly what good and bad look like.

### 4. Verification caching with content-addressing

LLM verify results are cached by MD5 hash of the full condition + state.
Re-running the same contract with unchanged content hits the cache, making
repeated verification fast and cost-free. This is crucial for development
loops where you're iterating on one contract while others remain stable.

### 5. Automatic tool-chain integration

Contract types integrate external tools automatically at construction time:

- `PythonFile` runs `pyright` and `ruff check` + `ruff format`
- `MarkdownFile` runs `markdownlint-cli2 --fix` + `markdownlint-cli2`
- `Directory` validates its `content` list matches actual disk files,
  respecting `.gitignore`

The contract doesn't just describe what should exist — it runs the tools to
prove it.

### 6. Self-aware content management

`File.content` is auto-populated from disk on first read, but if you
*provide* content manually and it doesn't match the file on disk,
construction fails. This creates a powerful pattern: contracts can carry
expected content as a field and the framework verifies reality matches
expectation.

### 7. Human-expert-review as a framework concept

`BaseContract.human_expert_review(reviewer, metric)` is a no-op method — but
its *presence* signals that programmatic and LLM verification are not the
final word. The framework explicitly acknowledges that expert human judgment
is the ultimate benchmark, and automated checks are preliminary gates.

### 8. Single-file reveal.js hosting

The `Reveal` contract wraps a `MarkdownFile` and can serve it as a full
reveal.js slide deck with a single method call
(`Reveal(markdown=...).host()`). It's a contract that doubles as a
deliverable — define the content, verify it, and present it, all from the
same object.

---

## Contract Hierarchy

```text
BaseContract               — llm_verify(), human_expert_review(),
                             pretty-printing
├── File                   — path + auto-filled content, os.path.exists()
│   ├── PythonFile         — + pyright, ruff check, ruff format
│   └── MarkdownFile       — + markdownlint-cli2 --fix + lint
├── Image                  — SVG/PNG/JPG/GIF/WEBP, auto-extracts
│                            dimensions & metadata
├── Directory              — path + File[], validates tree matches
│   │                        .gitignore-filtered reality
│   └── PythonWorkspace    — + pyright, ruff check, ruff format
├── Contract               — meta-contract that generates new classes
├── Concept                — semantic validation for documented ideas
└── Reveal                 — markdown-to-slides hosting
```

---

## How It Fits Together

1. **Design**: Define what your agent needs to produce as a `Contract` spec
   (fields + verify conditions with examples).
2. **Validate**: The meta-contract LLM-verifies that your examples truly
   pass/fail their conditions.
3. **Generate**: A contract class file is auto-generated with all checks
   embedded.
4. **Execute**: Your agent instantiates the contract with its output → all
   validations fire → either exception or success.
5. **Iterate**: Cached verifications make re-runs fast. Change the spec,
   regenerate, re-execute.
