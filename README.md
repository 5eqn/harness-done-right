# Harness Done Right

A magical Agent Harness that makes your agent **do things right**.

Inspired from [Curry-Howard Correspondence](https://en.wikipedia.org/wiki/Curry%E2%80%93Howard_correspondence), we developed a DSL (Domain Specific Language) for general agent tasks, and made a shallow embedding to the host language Python.

## Example: Humanizing Text

### 1. Task Definition
Your agent formalizes the requirement into a schema:

```python
from hdr import BaseModel, verify, quote

class HumanizeText(BaseModel):
    original: str
    humanized: str

    def __init__(self, **data):
        super().__init__(**data)
        # LLM-backed assertions
        verify(f"{quote(self.original)} and {quote(self.humanized)} convey the same meaning")
        verify(f"{quote(self.humanized)} reads like natural human-written text")
```

### 2. Execution
Your agent runs the script to perform and verify the task:

```python
# Set up environment variables:
# export ANTHROPIC_API_KEY="your-api-key"
# export ANTHROPIC_MODEL="claude-4.6-sonnet"  # optional, defaults to claude-4.6-sonnet

# Instantiation triggers Pydantic type checks and LLM verification
result = HumanizeText(
    original="AI-generated technical jargon...",
    humanized="A clear, human-friendly explanation..."
)
print("Task Verified:", result)
```

## Getting Started

### Environment Setup

HDR requires the following environment variables:
- `ANTHROPIC_API_KEY`: Your Anthropic API key (required)
- `ANTHROPIC_MODEL`: Model name (optional, defaults to claude-4.6-sonnet)
- `ANTHROPIC_BASE_URL`: API base URL (optional)

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

We're currently developing use cases for this repo.

## Development

```bash
uv venv .venv
uv pip install -e ".[dev]"
source .venv/bin/activate
```
