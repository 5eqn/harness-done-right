import os
import json
import subprocess
from typing import Any

from pydantic import BaseModel

# HDR directory configuration
HDR_DIR = os.path.expanduser("~/.hdr")
CONFIG_FILE = os.path.join(HDR_DIR, "config.json")

os.makedirs(HDR_DIR, exist_ok=True)

# Internal mock mode flag - for testing only
# NEVER use mock mode in production or real execution environments
_mock_mode = False

def set_mock_mode(enabled: bool) -> None:
    """
    Toggle mock mode for testing purposes.
    WARNING: Mock mode should ONLY be used in pytest unit tests.
    In mock mode, verify() returns mock results instead of calling Claude CLI.
    """
    global _mock_mode
    _mock_mode = enabled

def load_config():
    """Load configuration from ~/.hdr/config.json"""
    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    return config

def save_config(config):
    """Save configuration to ~/.hdr/config.json"""
    os.makedirs(HDR_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

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


def _verify(condition: str) -> tuple[str, int]:
    """Internal verify function that calls Claude Code CLI and returns thinking and score"""
    # Return mock result if mock mode is enabled
    if _mock_mode:
        return ("Mock reasoning", 5)

    schema = json.dumps({
        "type": "object",
        "properties": {
            "thinking": {
                "type": "string",
                "description": "Your reasoning about the condition"
            },
            "score": {
                "type": "integer",
                "description": "Score from 1 to 5 indicating how well the condition is satisfied (5=completely satisfied, 1=completely unsatisfied)"
            }
        },
        "required": ["thinking", "score"]
    })

    prompt = f"""Evaluate this condition:

{condition}

First, think carefully about the condition using first principles, considering both supporting and opposing arguments to ensure a fair and balanced evaluation.
Then, output a JSON object with your thinking and a score from 1 to 5 where 5 means completely satisfied and 1 means completely unsatisfied.
Only output a score of 5 if the condition is 100% true with no exceptions."""

    cmd = [
        "claude",
        prompt,
        "--bare",
        "-p",
        "--tools", "Read",
        "--allowedTools", "Read",
        "--permission-mode", "dontAsk",
        "--max-turns", "10",
        "--output-format", "json",
        "--json-schema", schema
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Claude Code CLI failed: {result.stderr}")

    output = json.loads(result.stdout)

    if "error" in output:
        raise RuntimeError(f"Claude Code error: {output['error']}")

    structured = output.get("structured_output", {})
    thinking = structured.get("thinking", "")
    score = structured.get("score", 0)

    return thinking, score


def verify(condition: str) -> None:
    """
    Validate a condition using Claude Code. Throws an error with explanation if validation fails.
    """
    thinking, score = _verify(condition)

    if score != 5:
        raise AssertionError(f"Verification failed with score {score}/5.\nThinking: {thinking}\nScore: {score}/5")

# Export all public functions
__all__ = [
    "verify",
    "quote",
    "BaseModel",
    "load_config",
    "save_config",
    "HDR_DIR",
    "CONFIG_FILE"
]
