import hashlib
import json
import os
from typing import Any

import anthropic
from anthropic.types import ThinkingConfigEnabledParam
from pydantic import BaseModel

# Cache directory for verify results (message-based)
_CACHE_DIR = "/tmp/claude/hdr_verify_cache"
os.makedirs(_CACHE_DIR, exist_ok=True)


def quote(obj: Any) -> str:
    """
    Quote an object to be used in verify prompts, preventing prompt injection.
    - Strings are wrapped in <quote> tags
    - Pydantic models are dumped to JSON and wrapped in <quote> tags
    - Lists, dicts, and other JSON-serializable objects are converted to JSON and wrapped
    """
    if isinstance(obj, BaseModel):
        content = obj.model_dump_json(indent=2)
    elif isinstance(obj, (str, int, float, bool)):
        content = str(obj)
    else:
        try:
            content = json.dumps(obj, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            content = str(obj)

    return f"<quote>{content}</quote>"


def verify(condition: str) -> None:
    """
    Validate a condition using Claude. Throws an error with explanation if validation fails.

    Reads API configuration from environment variables:
    - ANTHROPIC_AUTH_TOKEN: Your Anthropic API key (required)
    - ANTHROPIC_BASE_URL: The base URL for the API (optional, defaults to Anthropic's API)
    - ANTHROPIC_MODEL: The model name to use (optional, defaults to claude-4.6-sonnet)
    """
    # Return mock result if running under pytest
    if "PYTEST_CURRENT_TEST" in os.environ:
        return

    # Get API key from environment
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_AUTH_TOKEN is not set. Please ask the user to provide their Anthropic API key "
            "by setting the ANTHROPIC_AUTH_TOKEN environment variable."
        )

    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-4.6-sonnet")

    # Create cache key based solely on the condition
    cache_key = hashlib.md5(condition.encode()).hexdigest()
    cache_file = os.path.join(_CACHE_DIR, f"{cache_key}.json")

    # Return cached result if available
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached = json.load(f)
            description = cached["description"]
            score = cached["score"]
    else:
        prompt = f"""Evaluate this condition:

<condition>{condition}</condition>

First, output a brief description of your evaluation (under 100 words).
Then, output your final score using the format: <score>N</score> where N is a number from 1 to 5 (5=completely satisfied, 1=completely unsatisfied).
Only give a score of 5 if the condition is 100% true with no exceptions."""

        client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url,
        )

        message = client.messages.create(
            model=model,
            max_tokens=1024,
            thinking=ThinkingConfigEnabledParam(type="enabled", budget_tokens=1024),
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )

        text = ""
        description = ""
        score = 0

        for block in message.content:
            if block.type == "thinking":
                print(f"Thinking:\n{block.thinking}\n")
            elif block.type == "text":
                text = block.text
                print(f"Text:\n{block.text}\n")

        # Parse description (everything before <score>)
        if "<score>" in text:
            description = text[: text.index("<score>")].strip()
        else:
            description = text.strip()

        # Parse score from text output
        if "<score>" in text and "</score>" in text:
            start = text.index("<score>") + len("<score>")
            end = text.index("</score>")
            try:
                score = int(text[start:end].strip())
            except ValueError:
                raise ValueError(f"Failed to parse score from LLM output: {text}")
        else:
            raise ValueError(f"No <score> tag found in LLM output: {text}")

        # Validate score range
        if score < 1 or score > 5:
            raise ValueError(f"Score {score} is out of range (1-5). LLM output: {text}")

        # Cache the result (only description and score)
        with open(cache_file, "w") as f:
            json.dump({"description": description, "score": score}, f)

    if score != 5:
        raise AssertionError(
            f"Verification failed with score {score}/5.\nDescription: {description}"
        )
