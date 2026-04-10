"""
Standard task types for common use cases.

These tasks cover typical file operations, content validation, and transformations.
All tasks use relative paths by preference, as they are more portable and make
projects easier to share and version control.
"""

import hashlib
import json
import os
import re
from typing import Any, Sequence

import anthropic
from anthropic.types import ThinkingConfigEnabledParam
from hdr.config import load_verify_config
from pydantic import BaseModel, Field


class Example:
    """
    Lightweight stand-in for a BaseModel instance, for use with quote().
    Represents a named object with described fields and values, without
    requiring a full Pydantic model definition.
    """

    def __init__(self, class_name: str, fields: dict[str, tuple[Any, str]]):
        """
        Args:
            class_name: Display name for the object.
            fields: Ordered dict mapping field_name -> (value, description).
        """
        self.class_name = class_name
        self.fields = fields


def _quote_named_fields(
    class_name: str,
    fields: Sequence[tuple[str, Any, str | None]],
    indent: int,
) -> str:
    """
    Format a named object with fields.
    Shared by BaseModel and Example branches of quote().

    Args:
        class_name: Name to display.
        fields: List of (field_name, field_value, description_or_None).
        indent: Current indentation level.
    """
    indent_str = "  " * indent
    next_indent = indent + 1
    next_indent_str = "  " * next_indent

    result = [f"{indent_str}{class_name}("]

    for field_name, field_value, description in fields:
        field_desc = f" # {description}" if description else ""

        if isinstance(field_value, (BaseModel, Example)):
            value_str = quote(field_value, next_indent).lstrip()
        elif isinstance(field_value, dict):
            value_str = " {"
            if field_value:
                value_str += "\n"
                for k, v in field_value.items():
                    value_str += f"{next_indent_str}{repr(k)}: {quote(v, next_indent + 1).lstrip()},\n"
                value_str += f"{indent_str}  "
            value_str += "}"
        elif isinstance(field_value, (list, tuple)):
            value_str = " ["
            if field_value:
                value_str += "\n"
                for item in field_value:
                    value_str += (
                        f"{next_indent_str}{quote(item, next_indent + 1).lstrip()},\n"
                    )
                value_str += f"{indent_str}  "
            value_str += "]"
        elif isinstance(field_value, str):
            value_str = repr(field_value)
        else:
            value_str = str(field_value)

        result.append(f"{next_indent_str}{field_name} = {value_str}{field_desc}")

    result.append(f"{indent_str})")
    return "\n".join(result)


def quote(obj: Any, indent: int = 0) -> str:
    """
    Pretty quote an object for use in verify prompts.
    - Prints class name for BaseModel and Example instances
    - Recursively formats nested objects, dicts, and arrays
    - Includes field descriptions if available
    - Uses indentation for readability
    """
    indent_str = "  " * indent
    next_indent = indent + 1
    next_indent_str = "  " * next_indent

    if isinstance(obj, BaseModel):
        fields = [
            (name, getattr(obj, name), info.description)
            for name, info in obj.__class__.model_fields.items()
        ]
        return _quote_named_fields(obj.__class__.__name__, fields, indent)

    elif isinstance(obj, Example):
        fields = [(name, value, desc) for name, (value, desc) in obj.fields.items()]
        return _quote_named_fields(obj.class_name, fields, indent)

    elif isinstance(obj, dict):
        if not obj:
            return f"{indent_str}{{}}"
        result = [f"{indent_str}{{"]
        for k, v in obj.items():
            result.append(
                f"{next_indent_str}{repr(k)}: {quote(v, next_indent + 1).lstrip()},"
            )
        result.append(f"{indent_str}}}")
        return "\n".join(result)

    elif isinstance(obj, (list, tuple)):
        if not obj:
            return f"{indent_str}[]"
        bracket = "[" if isinstance(obj, list) else "("
        close_bracket = "]" if isinstance(obj, list) else ")"
        result = [f"{indent_str}{bracket}"]
        for item in obj:
            result.append(f"{next_indent_str}{quote(item, next_indent + 1).lstrip()},")
        result.append(f"{indent_str}{close_bracket}")
        return "\n".join(result)

    elif isinstance(obj, str):
        return f"{indent_str}{repr(obj)}"

    else:
        return f"{indent_str}{str(obj)}"


def _summarize_condition(condition: str, max_length: int = 63) -> str:
    """Normalize whitespace and trim long conditions for single-line logs."""
    normalized = re.sub(r"\s+", " ", condition).strip()
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


class Task(BaseModel):
    """
    Base class for all HDR tasks.
    Provides built-in verify method that automatically includes the task's pretty-printed state.
    """

    def _call_llm_with_retry(
        self,
        full_condition: str,
        api_key: str,
        base_url: str,
        model: str,
        max_retries: int = 10,
        verbose: bool = False,
    ) -> tuple[str, int]:
        """
        Call LLM with retry logic. Returns (description, score).
        Only retries on errors (network, parsing, etc.), not on score < 5.
        """
        prompt = f"""Evaluate this condition:

<condition>{full_condition}</condition>

First, output a brief description of your evaluation (under 100 words).
Then, output your final score using the format: <score>N</score>, N ranges from:

1 / Definitely false. Even if parts are true, the overall claim cannot reasonably be interpreted as true.
    Example: "The document clearly explains how LLMs work" (when it only covers Shannon's Theorem — related, but the main claim is plainly unmet)

2 / Probably false, but the condition is ambiguous enough that a different reasonable interpretation could make it true.
    Example: "The response is short" (it's 150 words — is that short?)

3 / Genuinely ambiguous. Cannot be evaluated without clarification.
    Example: "The output is good"

4 / Probably true, but the condition is ambiguous enough that a different reasonable interpretation could make it false.
    Example: "The code handles errors" (it handles some, but all?)

5 / Definitely true. Judge by standard interpretation; do not search for edge cases to invalidate a clearly true statement.
    Example: "The code contains no factual errors" (it says JavaScript arrays are zero-indexed — true under any reasonable reading)
"""

        for attempt in range(1, max_retries + 1):
            try:
                client = anthropic.Anthropic(
                    api_key=api_key,
                    base_url=base_url,
                )

                message = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    thinking=ThinkingConfigEnabledParam(
                        type="enabled", budget_tokens=1024
                    ),
                    messages=[
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ],
                )

                text = ""
                description = ""
                score = 0

                # Log LLM response
                for block in message.content:
                    if block.type == "thinking":
                        if verbose:
                            print(f"Thinking:\n{block.thinking}\n")
                    elif block.type == "text":
                        text = block.text
                        if verbose:
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
                        raise ValueError(
                            f"Failed to parse score from LLM output: {text}"
                        )
                else:
                    raise ValueError(f"No <score> tag found in LLM output: {text}")

                # Validate score range
                if score < 1 or score > 5:
                    raise ValueError(
                        f"Score {score} is out of range (1-5). LLM output: {text}"
                    )

                # Success - return the result
                return description, score

            except Exception as e:
                print(f"Error on attempt {attempt}/{max_retries}: {e}")
                if attempt == max_retries:
                    raise Exception(f"Failed after {max_retries} retries: {e}")
                # Continue to next retry

        # Should never reach here
        raise Exception("Unexpected end of retry loop")

    def verify(
        self, condition: str, expected_score: int = 5, inject_self_quote: bool = True
    ) -> None:
        """
        Verify a condition against the current task state.
        Automatically includes the pretty-printed task object as context.
        No need to manually quote fields or use f-strings.
        """
        if inject_self_quote:
            full_condition = f"Given the task object:\n{quote(self)}\n\nThis condition holds true: {condition}"
        else:
            full_condition = condition

        # Return mock result if running under pytest
        if "PYTEST_CURRENT_TEST" in os.environ:
            # Parse mock score from condition if present: <mock>N</mock>
            match = re.search(r"<mock>(\d)</mock>", condition)
            if match:
                score = int(match.group(1))
            else:
                score = 5
            if score != expected_score:
                raise AssertionError(
                    f"Mock verification failed with score {score} (expected {expected_score})"
                )
            print(f"[verify] score={score} {_summarize_condition(condition)}")
            return

        verify_config = load_verify_config()

        # Create cache key based solely on the condition
        cache_dir = verify_config.verify_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.md5(full_condition.encode()).hexdigest()
        cache_file = cache_dir / f"{cache_key}.json"

        # Return cached result if available
        if cache_file.exists():
            with cache_file.open("r") as f:
                cached = json.load(f)
                description = cached["description"]
                score = cached["score"]
        else:
            # Call LLM with retry logic
            description, score = self._call_llm_with_retry(
                full_condition,
                api_key=verify_config.anthropic_auth_token,
                base_url=verify_config.anthropic_base_url,
                model=verify_config.anthropic_model,
            )

            # Cache the result (only description and score)
            with cache_file.open("w") as f:
                json.dump({"description": description, "score": score}, f)

        if score != expected_score:
            raise AssertionError(
                f"Verification failed with score {score} (expected {expected_score}).\nDescription: {description}"
            )
        print(f"[verify] score={score} {_summarize_condition(condition)}")


class FileWritten(Task):
    """
    Validates that a file exists at the given path using os.path.exists().

    Prefer using relative paths for portability. The path can be absolute
    if needed, but relative paths are recommended for project-agnostic code.

    The `content` field is auto-filled from the actual file content and cannot be manually assigned.
    """

    path: str = Field(description="Path to the file")
    content: str = Field(
        init=False,
        default="",
        description="Content of the file, auto-filled from disk (cannot be manually assigned)",
    )

    def __init__(self, **data):
        super().__init__(**data)
        if not os.path.exists(self.path):
            raise AssertionError(f"File at {self.path} does not exist")
        # Always auto-fill content from actual file since it can't be passed
        try:
            with open(self.path, "r") as f:
                self.content = f.read()
        except (IOError, OSError):
            raise AssertionError(f"Could not read file at {self.path}")


class DirectoryCreated(Task):
    """
    Validates that a directory exists at the given path using os.path.isdir().

    The `content` field is auto-filled from the actual directory content if not specified.
    Content is a list of FileWritten objects representing the files in the directory,
    gathered recursively and respecting .gitignore patterns.
    """

    path: str = Field(description="Path to the directory")
    content: list[FileWritten] = Field(
        default_factory=list,
        description="List of FileWritten objects in the directory, auto-filled if not provided",
    )

    def __init__(self, **data):
        super().__init__(**data)
        if not os.path.isdir(self.path):
            raise AssertionError(f"Directory at {self.path} does not exist")
        # Auto-fill content from actual directory if not provided
        if not self.content:
            self.content = self._gather_content(self.path)
            total_files = len(self.content)
            print(f"[Directory] Total files in {self.path}: {total_files}")

    def _gather_content(self, dir_path: str) -> list[FileWritten]:
        """Gather content from directory as list[FileWritten], respecting .gitignore and recursing."""
        files: list[FileWritten] = []
        gitignore_path = os.path.join(dir_path, ".gitignore")
        gitignore_patterns = []
        if os.path.exists(gitignore_path):
            try:
                with open(gitignore_path, "r") as f:
                    gitignore_patterns = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    ]
            except (IOError, OSError):
                pass

        for root, dirs, filenames in os.walk(dir_path):
            # Calculate relative path for filtering directories
            rel_root = os.path.relpath(root, dir_path)
            # Filter out directories matching gitignore patterns
            dirs[:] = [
                d
                for d in dirs
                if not self._is_ignored(os.path.join(rel_root, d), gitignore_patterns)
            ]

            for filename in filenames:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, dir_path)
                # Check if file should be ignored
                if self._is_ignored(rel_path, gitignore_patterns):
                    continue
                try:
                    with open(filepath, "r") as f:
                        file_content = f.read()
                    files.append(FileWritten(path=filepath, content=file_content))
                except (IOError, OSError):
                    pass

        return files

    def _is_ignored(self, rel_path: str, patterns: list[str]) -> bool:
        """Check if a path matches any gitignore pattern."""
        import fnmatch

        for pattern in patterns:
            if pattern.endswith("/"):
                # Directory pattern
                dir_pattern = pattern.rstrip("/")
                if fnmatch.fnmatch(rel_path, dir_pattern) or fnmatch.fnmatch(
                    rel_path, dir_pattern + "/*"
                ):
                    return True
            else:
                if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(
                    os.path.basename(rel_path), pattern
                ):
                    return True
        return False
