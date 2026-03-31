import os
import json
import subprocess
import hashlib
from typing import Any

from pydantic import BaseModel

# Base directory for all HDR temporary files
_BASE_TMP_DIR = "/tmp/claude/hdr"
os.makedirs(_BASE_TMP_DIR, exist_ok=True)

# Global commit hash for checkout - determines where Claude Code runs
_current_commit: str = ""

# Internal checkout directory - stored for verify to access
_current_checkout_dir: str = ""

# Cache directory for verify results (commit-aware)
_CACHE_DIR = os.path.join(_BASE_TMP_DIR, "hdr_verify_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

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


def checkout(commit: str = "") -> None:
    """
    Set up the working directory for a given git commit.

    Uses git archive to extract the repository state at the given commit
    to {_BASE_TMP_DIR}/{commit}. If already extracted, uses the cached directory.

    The directory is stored internally and can be retrieved with get_checkout_dir().

    Args:
        commit: The git commit hash (empty string for no commit creates a unique temp directory)
    """
    global _current_commit, _current_checkout_dir

    if commit:
        _current_commit = commit
        target_dir = os.path.join(_BASE_TMP_DIR, commit)
    else:
        _current_commit = ""
        target_dir = os.path.join(_BASE_TMP_DIR, "hdr_no_commit")

    _current_checkout_dir = target_dir

    if os.path.exists(target_dir):
        return

    os.makedirs(target_dir, exist_ok=True)

    if commit:
        # Use git archive to extract the commit state
        result = subprocess.run(
            ["git", "archive", "--format", "tar", commit],
            capture_output=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        if result.returncode == 0:
            import tarfile
            import io
            with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r') as tar:
                tar.extractall(target_dir, filter=tarfile.data_filter)


def get_checkout_dir() -> str:
    """
    Get the current checkout directory.

    If checkout() has been called, returns that directory.
    Otherwise, creates a uniquely named empty directory.

    Returns:
        The path to the working directory
    """
    global _current_checkout_dir

    if _current_checkout_dir:
        return _current_checkout_dir

    # Create a unique temp directory if checkout hasn't been called
    import tempfile
    unique_dir = tempfile.mkdtemp(prefix="hdr_no_checkout_", dir=_BASE_TMP_DIR)
    _current_checkout_dir = unique_dir
    os.makedirs(unique_dir, exist_ok=True)
    return unique_dir


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

    # Create cache key that includes checkout directory for isolation across checkouts
    checkout_dir = get_checkout_dir()
    cache_key = hashlib.md5(f"{checkout_dir}:{condition}".encode()).hexdigest()
    cache_file = os.path.join(_CACHE_DIR, f"{cache_key}.json")

    # Return cached result if available
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached = json.load(f)
            return cached["thinking"], cached["score"]

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

    # Cache the result
    with open(cache_file, "w") as f:
        json.dump({"thinking": thinking, "score": score}, f)

    return thinking, score


def verify(condition: str) -> None:
    """
    Validate a condition using Claude Code. Throws an error with explanation if validation fails.
    """
    thinking, score = _verify(condition)

    if score != 5:
        raise AssertionError(f"Verification failed with score {score}/5.\nThinking: {thinking}\nScore: {score}/5")
