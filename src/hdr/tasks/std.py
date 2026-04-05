"""
Standard task types for common use cases.

These tasks cover typical file operations, content validation, and transformations.
All tasks use relative paths by preference, as they are more portable and make
projects easier to share and version control.
"""

from typing import Any
import hashlib
import json
import os
import subprocess
import anthropic
from anthropic.types import ThinkingConfigEnabledParam
from pydantic import BaseModel, Field


def quote(obj: Any, indent: int = 0) -> str:
    """
    Pretty quote an object for use in verify prompts.
    - Prints class name for BaseModel instances
    - Recursively formats nested BaseModels, dicts, and arrays
    - Includes field descriptions if available
    - Uses indentation for readability
    """
    indent_str = "  " * indent
    next_indent = indent + 1
    next_indent_str = "  " * next_indent

    if isinstance(obj, BaseModel):
        class_name = obj.__class__.__name__
        result = [f"{indent_str}{class_name}("]

        # Get model fields with descriptions
        for field_name, field_info in obj.model_fields.items():
            field_value = getattr(obj, field_name)
            description = field_info.description
            field_desc = f" # {description}" if description else ""

            # Format field value
            if isinstance(field_value, BaseModel):
                value_str = "\n" + quote(field_value, next_indent)
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
                        value_str += f"{next_indent_str}{quote(item, next_indent + 1).lstrip()},\n"
                    value_str += f"{indent_str}  "
                value_str += "]"
            elif isinstance(field_value, str):
                value_str = repr(field_value)
            else:
                value_str = str(field_value)

            result.append(f"{next_indent_str}{field_name} = {value_str}{field_desc}")

        result.append(f"{indent_str})")
        return "\n".join(result)

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


class Task(BaseModel):
    """
    Base class for all HDR tasks.
    Provides built-in verify method that automatically includes the task's pretty-printed state.
    """

    # Cache directory for verify results (message-based)
    _CACHE_DIR = "/tmp/claude/hdr_verify_cache"
    os.makedirs(_CACHE_DIR, exist_ok=True)

    def _call_llm_with_retry(
        self, full_condition: str, max_retries: int = 10
    ) -> tuple[str, int]:
        """
        Call LLM with retry logic. Returns (description, score).
        Only retries on errors (network, parsing, etc.), not on score < 5.
        """
        prompt = f"""Evaluate this condition:

<condition>{full_condition}</condition>

First, output a brief description of your evaluation (under 100 words).
Then, output your final score using the format: <score>N</score> where N is a number from 1 to 5 (5=completely satisfied, 1=completely unsatisfied).
Only give a score of 5 if the condition is 100% true with no exceptions."""

        api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        model = os.environ.get("ANTHROPIC_MODEL", "claude-4.6-sonnet")

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

    def verify(self, condition: str) -> None:
        """
        Verify a condition against the current task state.
        Automatically includes the pretty-printed task object as context.
        No need to manually quote fields or use f-strings.
        """
        full_condition = (
            f"Given the task object:\n{quote(self)}\n\nVerify that: {condition}"
        )

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

        # Create cache key based solely on the condition
        cache_key = hashlib.md5(full_condition.encode()).hexdigest()
        cache_file = os.path.join(self._CACHE_DIR, f"{cache_key}.json")

        # Return cached result if available
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cached = json.load(f)
                description = cached["description"]
                score = cached["score"]
        else:
            # Call LLM with retry logic
            description, score = self._call_llm_with_retry(full_condition)

            # Cache the result (only description and score)
            with open(cache_file, "w") as f:
                json.dump({"description": description, "score": score}, f)

        if score != 5:
            raise AssertionError(
                f"Verification failed with score {score}/5.\nDescription: {description}"
            )


class FileWritten(Task):
    """
    Validates that a file exists at the given path using os.path.exists().

    Prefer using relative paths for portability. The path can be absolute
    if needed, but relative paths are recommended for project-agnostic code.

    The `content` field is auto-filled from the actual file content if not specified.
    """

    path: str = Field(description="Path to the file")
    content: str = Field(
        default="", description="Content of the file, auto-filled if not provided"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if not os.path.exists(self.path):
            raise AssertionError(f"File at {self.path} does not exist")
        # Auto-fill content from actual file if not provided
        if not self.content:
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


class MarkdownFileWritten(FileWritten):
    """
    Validates that a markdown file exists at the given path.
    Inherits all fields from FileWritten.

    Additionally verifies:
    - Path ends with `.md`
    - markdownlint-cli2 reports no syntax errors
    """

    def __init__(self, **data):
        super().__init__(**data)
        if not self.path.endswith(".md"):
            raise AssertionError(f"Path '{self.path}' does not end with '.md'")
        result = subprocess.run(
            ["markdownlint-cli2", self.path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"markdownlint-cli2 found issues in {self.path}:\n{result.stderr}\n{result.stdout}"
            )


class ConceptDescribed(Task):
    """
    Represents a documented concept within a context.
    """

    context: MarkdownFileWritten = Field(
        description="File explaining the parent context"
    )
    name: str = Field(description="Name of the concept")
    description: MarkdownFileWritten = Field(
        description="File containing the concept description"
    )

    def __init__(self, **data):
        super().__init__(**data)

        self.verify(
            "The description is written for readers who understand context but do not yet know name; it neither repeats basics from context nor presumes knowledge of sibling/descendant concepts."
        )
        self.verify(
            "The concept name represents exactly one atomic idea that cannot be meaningfully split into two independent concepts."
        )
        self.verify(
            "The description contains no time-sensitive terms (e.g., 'currently', 'recently', 'as of now') without specifying an exact version or date."
        )
        self.verify(
            "The description identifies (a) a broader category that name belongs to, and (b) a distinguishing property that separates it from other members of that category."
        )
        self.verify(
            "A reader familiar with context can determine for any concrete instance whether it belongs to name, with at most minor edge-case ambiguity."
        )
