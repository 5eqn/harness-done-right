"""
Standard contract types for common use cases.

These contracts cover typical file operations, content validation, and transformations.
All contracts use relative paths by preference, as they are more portable and make
projects easier to share and version control.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Sequence

import anthropic
from anthropic.types import ThinkingConfigEnabledParam
from hdr.config import load_verify_config
from pydantic import BaseModel, Field, model_validator


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


class BaseContract(BaseModel):
    """
    Base class for all HDR contracts.
    Provides built-in verify method that automatically includes the contract's pretty-printed state.
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
        print(f"[verify] score={score} {_summarize_condition(condition)}", flush=True)


@dataclass(frozen=True, slots=True)
class _GitignoreRule:
    base_rel: str
    pattern: str
    directory_only: bool


class File(BaseContract):
    """
    Validates that a file exists at the given path using os.path.exists().

    Prefer using relative paths for portability. The path can be absolute
    if needed, but relative paths are recommended for project-agnostic code.

    The `content` field is auto-filled from the actual file content and cannot be manually assigned.
    """

    path: str = Field(description="Path to the file")
    content: str = Field(
        init=False,
        frozen=True,
        default="",
        description="Content of the file, auto-filled from disk (cannot be manually assigned)",
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_content(cls, data: Any) -> Any:
        """Populate frozen content from disk before model construction."""
        if not isinstance(data, dict):
            return data

        data = dict(data)
        data.pop("content", None)
        path = data.get("path")
        if not isinstance(path, str):
            return data
        if not os.path.exists(path):
            raise AssertionError(f"File at {path} does not exist")

        content = data.pop("_hdr_content", None)
        if content is None:
            content = cls._read_file_content(path)

        data["content"] = content
        return data

    def __init__(self, **data):
        path = data.get("path")
        if isinstance(path, str) and not os.path.exists(path):
            raise AssertionError(f"File at {path} does not exist")
        super().__init__(**data)
        if not os.path.exists(self.path):
            raise AssertionError(f"File at {self.path} does not exist")

    @staticmethod
    def _read_file_content(path: str) -> str:
        try:
            with open(path, "r") as f:
                return f.read()
        except (IOError, OSError):
            raise AssertionError(f"Could not read file at {path}")


class Directory(BaseContract):
    """
    Validates that a directory exists at the given path using os.path.isdir().

    The `content` field is auto-filled from the actual directory content if not specified.
    Content is a list of File objects representing the files in the directory,
    gathered recursively and respecting .gitignore patterns.
    """

    gather_content_on_init: ClassVar[bool] = True

    path: str = Field(description="Path to the directory")
    content: list[File] = Field(
        default_factory=list,
        description="List of File objects in the directory, auto-filled if not provided",
    )

    def __init__(self, **data):
        super().__init__(**data)
        if not os.path.isdir(self.path):
            raise AssertionError(f"Directory at {self.path} does not exist")
        # Auto-fill content from actual directory if not provided
        if self.gather_content_on_init and not self.content:
            self.content = self._gather_content(self.path)
            total_files = len(self.content)
            print(f"[Directory] Total files in {self.path}: {total_files}")

    def _gather_content(self, dir_path: str) -> list[File]:
        """Gather content from directory as list[File], respecting .gitignore and recursing."""
        files: list[File] = []
        gitignore_rules: list[_GitignoreRule] = []

        for root, dirs, filenames in os.walk(dir_path):
            self._extend_gitignore_rules(dir_path, root, gitignore_rules)

            # Calculate relative path for filtering directories
            rel_root = os.path.relpath(root, dir_path)
            # Filter out directories matching gitignore patterns
            dirs[:] = [
                d
                for d in dirs
                if not self._is_ignored(
                    self._join_rel(rel_root, d), gitignore_rules, is_dir=True
                )
            ]

            for filename in filenames:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, dir_path)
                # Check if file should be ignored
                if self._is_ignored(rel_path, gitignore_rules, is_dir=False):
                    continue
                try:
                    files.append(File(path=filepath))
                except (IOError, OSError):
                    pass

        return files

    def _extend_gitignore_rules(
        self, dir_path: str, root: str, rules: list[_GitignoreRule]
    ) -> None:
        """Load .gitignore rules scoped to the current walked directory."""
        gitignore_path = os.path.join(root, ".gitignore")
        if not os.path.exists(gitignore_path):
            return

        base_rel = os.path.relpath(root, dir_path)
        if base_rel == ".":
            base_rel = ""

        try:
            with open(gitignore_path, "r") as f:
                for line in f:
                    pattern = line.strip()
                    if (
                        not pattern
                        or pattern.startswith("#")
                        or pattern.startswith("!")
                    ):
                        continue
                    directory_only = pattern.endswith("/")
                    rules.append(
                        _GitignoreRule(
                            base_rel=base_rel,
                            pattern=pattern.strip("/"),
                            directory_only=directory_only,
                        )
                    )
        except (IOError, OSError):
            pass

    def _is_ignored(
        self, rel_path: str, rules: list[_GitignoreRule], is_dir: bool
    ) -> bool:
        """Check if a path matches any gitignore pattern."""
        import fnmatch

        normalized_path = rel_path.replace(os.sep, "/")

        for rule in rules:
            if rule.directory_only and not is_dir:
                continue

            scoped_path = self._path_relative_to_rule_base(
                normalized_path, rule.base_rel
            )
            if scoped_path is None:
                continue

            pattern = rule.pattern
            if "/" in pattern:
                if fnmatch.fnmatch(scoped_path, pattern):
                    return True
            elif fnmatch.fnmatch(os.path.basename(scoped_path), pattern):
                return True
        return False

    @staticmethod
    def _join_rel(rel_root: str, name: str) -> str:
        if rel_root == ".":
            return name
        return os.path.join(rel_root, name)

    @staticmethod
    def _path_relative_to_rule_base(rel_path: str, base_rel: str) -> str | None:
        if not base_rel:
            return rel_path
        base_rel = base_rel.replace(os.sep, "/")
        if rel_path == base_rel:
            return ""
        prefix = f"{base_rel}/"
        if rel_path.startswith(prefix):
            return rel_path[len(prefix) :]
        return None
