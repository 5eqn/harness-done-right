"""
Meta-task for creating well-formed HDR tasks.

TaskCreated validates task definitions at construction time and generates
complete task class code with examples embedded in verify conditions.
"""

import re
from typing import Any
from pydantic import Field, BaseModel
from hdr.tasks.std import Task, Example, quote
from hdr.tasks.coding import PythonFileWritten


class FieldSpec(BaseModel):
    """Specification for a single field in a task."""

    name: str = Field(description="Name of the field")
    type_annotation: str = Field(
        description="Type annotation as a string (e.g., 'str', 'list[int]', 'FileWritten')"
    )
    description: str = Field(
        description="Human-readable description of what this field represents"
    )
    default: str | None = Field(
        default=None,
        description="Default value as a string expression, or None if required",
    )


class VerifySpec(BaseModel):
    """Specification for a single verify call with required examples."""

    condition: str = Field(description="The condition text to verify")
    positive_example: dict[str, Any] = Field(
        description="Field values that should score 5 (definitely true)"
    )
    negative_example: dict[str, Any] = Field(
        description="Field values that should score 1 (definitely false)"
    )


class TaskCreated(Task):
    """
    Meta-task for creating well-formed HDR tasks.

    Validates that:
    - All fields have descriptions (programmatic)
    - All verify specs have valid positive and negative examples (LLM-verified at instantiation)

    Provides code generation that embeds validated examples into verify conditions.
    Automatically generates a PythonFileWritten task for the created task class,
    ensuring the file passes PyRight and Ruff checks.
    """

    class_name: str = Field(description="Name of the task class to create")
    parent_class: str = Field(
        default="Task", description="Parent class to inherit from"
    )
    docstring: str = Field(description="Docstring for the task class")
    imports: list[str] = Field(
        default_factory=lambda: [
            "from pydantic import Field",
            "from hdr.tasks.std import Task",
        ],
        description="List of import statements to include at the top of the file",
    )
    fields: list[FieldSpec] = Field(description="List of field specifications")
    programmatic_checks: list[str] = Field(
        default_factory=list,
        description="Python code snippets for programmatic validations, executed before LLM verifies",
    )
    verifies: list[VerifySpec] = Field(
        description="List of verify specifications with examples"
    )
    generated_file: PythonFileWritten | None = Field(
        init=False,
        default=None,
        description="The generated Python file task instance, after successful creation (cannot be manually assigned). Will always be non-None after successful initialization.",
    )

    def __init__(self, **data):
        super().__init__(**data)

        # Programmatic: class_name must be valid Python identifier
        if not self.class_name.isidentifier():
            raise AssertionError(
                f"Class name '{self.class_name}' is not a valid Python identifier."
            )

        # Programmatic: all fields must have descriptions
        for field_spec in self.fields:
            if not field_spec.description or not field_spec.description.strip():
                raise AssertionError(
                    f"Field '{field_spec.name}' is missing a description. "
                    f"All fields must have descriptions for HDR tasks."
                )

        # Programmatic: field names must be valid Python identifiers
        for field_spec in self.fields:
            if not field_spec.name.isidentifier():
                raise AssertionError(
                    f"Field name '{field_spec.name}' is not a valid Python identifier."
                )

        # Programmatic: examples must reference correct fields
        required_fields = {f.name for f in self.fields if f.default is None}
        all_fields = {f.name for f in self.fields}

        for i, verify_spec in enumerate(self.verifies):
            for label, example in [
                ("positive_example", verify_spec.positive_example),
                ("negative_example", verify_spec.negative_example),
            ]:
                provided = set(example.keys())
                missing = required_fields - provided
                if missing:
                    raise AssertionError(
                        f"Verify #{i + 1} {label} is missing required fields: {missing}"
                    )
                extra = provided - all_fields
                if extra:
                    raise AssertionError(
                        f"Verify #{i + 1} {label} has unknown fields: {extra}"
                    )

        # Programmatic: generate the Python file and validate it with PythonFileWritten
        file_content = self._generate_file_content()
        file_path = self._generate_file_path()
        with open(file_path, "w") as f:
            f.write(file_content)
        self.generated_file = PythonFileWritten(path=file_path)

        # LLM verification: validate each example against its condition
        for i, verify_spec in enumerate(self.verifies):
            pos_quoted = quote(self._to_example(verify_spec.positive_example))
            neg_quoted = quote(self._to_example(verify_spec.negative_example))

            pos_condition = (
                f"Given the task object:\n{pos_quoted}\n\n"
                f"This condition holds true: {verify_spec.condition}"
            )
            self.verify(pos_condition, expected_score=5, inject_self_quote=False)

            neg_condition = (
                f"Given the task object:\n{neg_quoted}\n\n"
                f"This condition holds true: {verify_spec.condition}"
            )
            self.verify(neg_condition, expected_score=1, inject_self_quote=False)

    def _to_example(self, values: dict[str, Any]) -> Example:
        """Convert a dict of field values into an Example for quoting."""
        field_descs = {f.name: f.description for f in self.fields}
        return Example(
            class_name=self.class_name,
            fields={
                name: (value, field_descs.get(name, ""))
                for name, value in values.items()
            },
        )

    def _camel_to_snake(self, s: str) -> str:
        """Convert CamelCase to snake_case."""
        s = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s).lower()

    def _generate_file_content(self) -> str:
        """Generate the complete Python file content including imports and class code."""
        lines = []
        # Add imports
        for import_line in self.imports:
            lines.append(import_line)
        lines.append("")
        # Add class code
        lines.append(self._generate_code())
        return "\n".join(lines)

    def _generate_file_path(self) -> str:
        """Generate the snake_case .py filename from the class name."""
        return f"{self._camel_to_snake(self.class_name)}.py"

    def _generate_code(self) -> str:
        """
        Generate the complete Python code for the task class.

        Programmatic checks come first, then LLM verifies with embedded examples.
        """
        lines = []

        # Class definition
        lines.append(f"class {self.class_name}({self.parent_class}):")

        # Docstring
        if self.docstring:
            docstring_lines = self.docstring.strip().split("\n")
            if len(docstring_lines) == 1:
                lines.append(f'    """{docstring_lines[0]}"""')
            else:
                lines.append('    """')
                for doc_line in docstring_lines:
                    lines.append(f"    {doc_line}")
                lines.append('    """')
        lines.append("")

        # Fields
        for field_spec in self.fields:
            if field_spec.default is not None:
                lines.append(
                    f"    {field_spec.name}: {field_spec.type_annotation} = "
                    f'Field(default={field_spec.default}, description="{_escape(field_spec.description)}")'
                )
            else:
                lines.append(
                    f"    {field_spec.name}: {field_spec.type_annotation} = "
                    f'Field(description="{_escape(field_spec.description)}")'
                )
        lines.append("")

        # __init__
        lines.append("    def __init__(self, **data):")
        lines.append("        super().__init__(**data)")

        has_content = bool(self.programmatic_checks) or bool(self.verifies)

        if not has_content:
            lines.append("        pass")
        else:
            # Programmatic checks first (faster validation)
            for check in self.programmatic_checks:
                for check_line in check.strip().split("\n"):
                    lines.append(f"        {check_line}")

            # LLM verifies with embedded examples
            for verify_spec in self.verifies:
                condition_with_examples = self._build_condition_with_examples(
                    verify_spec
                )
                escaped = _escape_multiline(condition_with_examples)
                lines.append(f"        self.verify({escaped})")

        return "\n".join(lines)

    def _build_condition_with_examples(self, verify_spec: VerifySpec) -> str:
        """Build a condition string with embedded examples."""
        pos_quoted = quote(self._to_example(verify_spec.positive_example))
        neg_quoted = quote(self._to_example(verify_spec.negative_example))

        return (
            f"{verify_spec.condition}\n\n"
            f"Example that PASSES (score 5):\n{pos_quoted}\n\n"
            f"Example that FAILS (score 1):\n{neg_quoted}"
        )


def _escape(s: str) -> str:
    """Escape a string for use inside a double-quoted Python string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _escape_multiline(s: str) -> str:
    """Escape a string, using triple quotes if it contains newlines."""
    if "\n" in s:
        escaped = s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        return f'"""{escaped}"""'
    else:
        return f'"{_escape(s)}"'
