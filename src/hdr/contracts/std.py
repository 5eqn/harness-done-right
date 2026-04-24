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

    def llm_verify(
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
    pattern: str
    directory_only: bool


class File(BaseContract):
    """
    Validates that a file exists at the given path using os.path.exists().

    Prefer using relative paths for portability. The path can be absolute
    if needed, but relative paths are recommended for project-agnostic code.

    The `content` field is auto-filled from disk when omitted. If it is provided
    manually, it must exactly match the file content on disk.
    """

    path: str = Field(description="Path to the file")
    content: str = Field(
        init=False,
        frozen=True,
        default="",
        description="Content of the file, auto-filled from disk when omitted, or validated against disk when provided",
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_content(cls, data: Any) -> Any:
        """Populate content from disk and validate manual content when provided."""
        if not isinstance(data, dict):
            return data

        data = dict(data)
        path = data.get("path")
        if not isinstance(path, str):
            return data
        if not os.path.exists(path):
            raise AssertionError(f"File at {path} does not exist")

        actual_content = data.pop("_hdr_content", None)
        if actual_content is None:
            actual_content = cls._read_file_content(path)

        expected_content = data.pop("content", None)
        if expected_content is not None and expected_content != actual_content:
            raise AssertionError(f"Content of {path} does not match the file on disk")

        data["content"] = (
            actual_content if expected_content is None else expected_content
        )
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


class Image(BaseContract):
    """
    Validates that an image exists and records render-oriented metadata.

    SVG is preferred: its text content is auto-filled so agents can inspect the
    diagram. Bitmap images are supported without reading binary bytes into a
    text field; their content is intentionally empty while media type, size, and
    dimensions are recorded when the header can be parsed.
    """

    supported_extensions: ClassVar[dict[str, str]] = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }

    path: str = Field(description="Path to the image file")
    media_type: str = Field(
        init=False,
        frozen=True,
        default="",
        description="Image media type inferred from file extension",
    )
    is_vector: bool = Field(
        init=False,
        frozen=True,
        default=False,
        description="True for SVG images, false for bitmap images",
    )
    width: int | None = Field(
        init=False,
        frozen=True,
        default=None,
        description="Image width in pixels when it can be inferred",
    )
    height: int | None = Field(
        init=False,
        frozen=True,
        default=None,
        description="Image height in pixels when it can be inferred",
    )
    size_bytes: int = Field(
        init=False,
        frozen=True,
        default=0,
        description="Image file size in bytes",
    )
    content: str = Field(
        init=False,
        frozen=True,
        default="",
        description="SVG text content; empty for bitmap images",
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_metadata(cls, data: Any) -> Any:
        """Populate image metadata before construction."""
        if not isinstance(data, dict):
            return data

        data = dict(data)
        for field_name in (
            "media_type",
            "is_vector",
            "width",
            "height",
            "size_bytes",
            "content",
        ):
            data.pop(field_name, None)

        path = data.get("path")
        if not isinstance(path, str):
            return data
        if not os.path.exists(path):
            raise AssertionError(f"Image at {path} does not exist")

        extension = cls._extension(path)
        media_type = cls.supported_extensions.get(extension)
        if media_type is None:
            supported = ", ".join(sorted(cls.supported_extensions))
            raise AssertionError(f"Image path must end with one of: {supported}")

        content = ""
        width: int | None = None
        height: int | None = None
        if extension == ".svg":
            content = cls._read_svg_content(path)
            width, height = cls._parse_svg_dimensions(content)
        else:
            width, height = cls._parse_bitmap_dimensions(path, extension)

        data["media_type"] = media_type
        data["is_vector"] = extension == ".svg"
        data["width"] = width
        data["height"] = height
        data["size_bytes"] = os.path.getsize(path)
        data["content"] = content
        return data

    def __init__(self, **data):
        path = data.get("path")
        if isinstance(path, str):
            if not os.path.exists(path):
                raise AssertionError(f"Image at {path} does not exist")
            if self._extension(path) not in self.supported_extensions:
                supported = ", ".join(sorted(self.supported_extensions))
                raise AssertionError(f"Image path must end with one of: {supported}")
        super().__init__(**data)
        if not os.path.exists(self.path):
            raise AssertionError(f"Image at {self.path} does not exist")

    @staticmethod
    def _extension(path: str) -> str:
        return os.path.splitext(path)[1].lower()

    @staticmethod
    def _read_svg_content(path: str) -> str:
        try:
            with open(path, "r") as f:
                return f.read()
        except (IOError, OSError):
            raise AssertionError(f"Could not read SVG image at {path}")

    @classmethod
    def _parse_svg_dimensions(cls, content: str) -> tuple[int | None, int | None]:
        width = cls._parse_svg_length(cls._svg_attr(content, "width"))
        height = cls._parse_svg_length(cls._svg_attr(content, "height"))
        if width is not None and height is not None:
            return width, height

        view_box = cls._svg_attr(content, "viewBox")
        if view_box is None:
            return width, height
        values = re.split(r"[\s,]+", view_box.strip())
        if len(values) != 4:
            return width, height
        try:
            return width or round(float(values[2])), height or round(float(values[3]))
        except ValueError:
            return width, height

    @staticmethod
    def _svg_attr(content: str, name: str) -> str | None:
        match = re.search(rf"\b{name}\s*=\s*['\"]([^'\"]+)['\"]", content)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _parse_svg_length(value: str | None) -> int | None:
        if value is None or value.strip().endswith("%"):
            return None
        match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", value)
        if not match:
            return None
        return round(float(match.group(1)))

    @classmethod
    def _parse_bitmap_dimensions(
        cls, path: str, extension: str
    ) -> tuple[int | None, int | None]:
        try:
            with open(path, "rb") as f:
                header = f.read(65536)
        except (IOError, OSError):
            raise AssertionError(f"Could not read image at {path}")

        if extension == ".png":
            return cls._parse_png_dimensions(header)
        if extension == ".gif":
            return cls._parse_gif_dimensions(header)
        if extension in {".jpg", ".jpeg"}:
            return cls._parse_jpeg_dimensions(header)
        if extension == ".webp":
            return cls._parse_webp_dimensions(header)
        return None, None

    @staticmethod
    def _parse_png_dimensions(header: bytes) -> tuple[int | None, int | None]:
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            return int.from_bytes(header[16:20], "big"), int.from_bytes(
                header[20:24], "big"
            )
        return None, None

    @staticmethod
    def _parse_gif_dimensions(header: bytes) -> tuple[int | None, int | None]:
        if header[:6] in {b"GIF87a", b"GIF89a"} and len(header) >= 10:
            return int.from_bytes(header[6:8], "little"), int.from_bytes(
                header[8:10], "little"
            )
        return None, None

    @staticmethod
    def _parse_jpeg_dimensions(header: bytes) -> tuple[int | None, int | None]:
        if not header.startswith(b"\xff\xd8"):
            return None, None

        i = 2
        start_of_frame_markers = {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }
        while i + 9 < len(header):
            if header[i] != 0xFF:
                i += 1
                continue
            while i < len(header) and header[i] == 0xFF:
                i += 1
            if i >= len(header):
                break
            marker = header[i]
            i += 1
            if marker in {0xD8, 0xD9}:
                continue
            if i + 2 > len(header):
                break
            segment_length = int.from_bytes(header[i : i + 2], "big")
            if segment_length < 2:
                break
            if marker in start_of_frame_markers and i + 7 < len(header):
                height = int.from_bytes(header[i + 3 : i + 5], "big")
                width = int.from_bytes(header[i + 5 : i + 7], "big")
                return width, height
            i += segment_length
        return None, None

    @staticmethod
    def _parse_webp_dimensions(header: bytes) -> tuple[int | None, int | None]:
        if (
            len(header) < 30
            or not header.startswith(b"RIFF")
            or header[8:12] != b"WEBP"
        ):
            return None, None
        chunk_type = header[12:16]
        if chunk_type == b"VP8X" and len(header) >= 30:
            width = int.from_bytes(header[24:27], "little") + 1
            height = int.from_bytes(header[27:30], "little") + 1
            return width, height
        if chunk_type == b"VP8 " and len(header) >= 30:
            width = int.from_bytes(header[26:28], "little") & 0x3FFF
            height = int.from_bytes(header[28:30], "little") & 0x3FFF
            return width, height
        if chunk_type == b"VP8L" and len(header) >= 25:
            bits = int.from_bytes(header[21:25], "little")
            width = (bits & 0x3FFF) + 1
            height = ((bits >> 14) & 0x3FFF) + 1
            return width, height
        return None, None


class Directory(BaseContract):
    """
    Validates that a directory exists at the given path using os.path.isdir().

    The `content` field must be manually assigned. It is validated against the
    actual directory's immediate files while respecting `.gitignore` patterns,
    and must match exactly with no missing or extra files.
    """

    path: str = Field(description="Path to the directory")
    content: list[File] = Field(
        default_factory=list,
        description="List of File objects whose paths must exactly match the directory contents after .gitignore filtering",
    )

    def __init__(self, **data):
        if "content" not in data:
            raise AssertionError(
                "Directory content must be manually assigned and match the directory"
            )
        super().__init__(**data)
        if not os.path.isdir(self.path):
            raise AssertionError(f"Directory at {self.path} does not exist")
        self._validate_tree_matches_directory()

    def _gather_content(self, dir_path: str) -> list[File]:
        """Gather immediate file content from a directory, respecting .gitignore."""
        files: list[File] = []
        gitignore_rules = self._load_gitignore_rules(dir_path)

        for entry in os.scandir(dir_path):
            if not entry.is_file():
                continue
            if self._is_ignored(entry.name, gitignore_rules, is_dir=False):
                continue
            try:
                files.append(File(path=entry.path))
            except (IOError, OSError):
                pass

        return files

    def _load_gitignore_rules(self, dir_path: str) -> list[_GitignoreRule]:
        """Load .gitignore rules from the current directory only."""
        gitignore_path = os.path.join(dir_path, ".gitignore")
        if not os.path.exists(gitignore_path):
            return []

        rules: list[_GitignoreRule] = []

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
                            pattern=pattern.strip("/"),
                            directory_only=directory_only,
                        )
                    )
        except (IOError, OSError):
            return []

        return rules

    def _is_ignored(
        self, rel_path: str, rules: list[_GitignoreRule], is_dir: bool
    ) -> bool:
        """Check if a path matches any gitignore pattern."""
        import fnmatch

        normalized_path = rel_path.replace(os.sep, "/")

        for rule in rules:
            if rule.directory_only and not is_dir:
                continue

            pattern = rule.pattern
            if "/" in pattern:
                if fnmatch.fnmatch(normalized_path, pattern):
                    return True
            elif fnmatch.fnmatch(os.path.basename(normalized_path), pattern):
                return True
        return False

    def _validate_tree_matches_directory(self) -> None:
        actual_files = self._gather_content(self.path)
        actual_paths = self._relpaths_from_content(actual_files)
        provided_paths = self._relpaths_from_content(self.content)

        missing_paths = sorted(actual_paths - provided_paths)
        extra_paths = sorted(provided_paths - actual_paths)
        if not missing_paths and not extra_paths:
            return

        details: list[str] = []
        if missing_paths:
            details.append(f"missing files: {', '.join(missing_paths)}")
        if extra_paths:
            details.append(f"unexpected files: {', '.join(extra_paths)}")
        raise AssertionError(
            "Directory content must exactly match the directory tree after .gitignore filtering ("
            + "; ".join(details)
            + ")"
        )

    def _relpaths_from_content(self, files: Sequence[File]) -> set[str]:
        directory_path = os.path.abspath(self.path)
        relpaths: set[str] = set()

        for file in files:
            file_path = os.path.abspath(file.path)
            rel_path = os.path.relpath(file_path, directory_path)
            if rel_path == "." or rel_path.startswith(".."):
                raise AssertionError(
                    f"Directory content file {file.path} is not inside {self.path}"
                )

            normalized_rel_path = rel_path.replace(os.sep, "/")
            if normalized_rel_path in relpaths:
                raise AssertionError(
                    f"Directory content contains duplicate file entry for {normalized_rel_path}"
                )
            relpaths.add(normalized_rel_path)

        return relpaths
