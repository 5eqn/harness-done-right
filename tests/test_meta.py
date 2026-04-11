"""
Tests for TaskCreated meta-task.

In pytest mode, verify() returns mock score 5 by default, or parses
<mock>N</mock> from the condition string. For TaskCreated to pass verify
in tests, negative examples embed <mock>1</mock> in a string field value
so that neg_condition gets score 1 (matching expected_score=1).
"""

import pytest
import os
from pydantic import ValidationError
from hdr.tasks.meta import AUTO_GENERATED_FLAG, TaskCreated, FieldSpec, VerifySpec


def _valid_fields() -> list[FieldSpec]:
    return [
        FieldSpec(name="input_text", type_annotation="str", description="Input text"),
        FieldSpec(name="output", type_annotation="str", description="Output text"),
    ]


def _valid_verifies() -> list[VerifySpec]:
    return [
        VerifySpec(
            condition="output is derived from input_text",
            positive_example={"input_text": "hello", "output": "HELLO"},
            negative_example={
                "input_text": "hello <mock>1</mock>",
                "output": "goodbye",
            },
        )
    ]


class TestTaskCreated:
    """Tests for TaskCreated meta-task."""

    def teardown_method(self):
        """Clean up generated files after each test."""
        generated_files = [
            "my_task.py",
            "summarized.py",
            "multi_task.py",
            "conditional_task.py",
        ]
        for file in generated_files:
            if os.path.exists(file):
                os.remove(file)

    def test_valid_construction(self):
        """Valid TaskCreated passes all programmatic and mock-verify checks."""
        task = TaskCreated(
            class_name="MyTask",
            docstring="A simple task.",
            fields=_valid_fields(),
            verifies=_valid_verifies(),
        )
        assert task.class_name == "MyTask"
        assert task.parent_class == "Task"
        assert len(task.fields) == 2
        assert len(task.verifies) == 1

    def test_invalid_class_name(self):
        """Class name must be a valid Python identifier."""
        with pytest.raises(AssertionError, match="not a valid Python identifier"):
            TaskCreated(
                class_name="123-invalid",
                docstring="Bad name.",
                fields=_valid_fields(),
                verifies=_valid_verifies(),
            )

    def test_field_missing_description(self):
        """All fields must have non-empty descriptions."""
        with pytest.raises(AssertionError, match="missing a description"):
            TaskCreated(
                class_name="MyTask",
                docstring="A task.",
                fields=[
                    FieldSpec(name="x", type_annotation="str", description=""),
                ],
                verifies=[
                    VerifySpec(
                        condition="x is valid",
                        positive_example={"x": "ok"},
                        negative_example={"x": "<mock>1</mock>"},
                    )
                ],
            )

    def test_invalid_field_name(self):
        """Field names must be valid Python identifiers."""
        with pytest.raises(AssertionError, match="not a valid Python identifier"):
            TaskCreated(
                class_name="MyTask",
                docstring="A task.",
                fields=[
                    FieldSpec(
                        name="bad-name",
                        type_annotation="str",
                        description="A field",
                    ),
                ],
                verifies=[
                    VerifySpec(
                        condition="field is valid",
                        positive_example={"bad-name": "ok"},
                        negative_example={"bad-name": "<mock>1</mock>"},
                    )
                ],
            )

    def test_example_missing_required_field(self):
        """Verify examples must include all required fields."""
        with pytest.raises(AssertionError, match="missing required fields"):
            TaskCreated(
                class_name="MyTask",
                docstring="A task.",
                fields=_valid_fields(),
                verifies=[
                    VerifySpec(
                        condition="output matches",
                        positive_example={"input_text": "hello"},
                        negative_example={"input_text": "hello", "output": "bad"},
                    )
                ],
            )

    def test_example_extra_field(self):
        """Verify examples must not contain unknown fields."""
        with pytest.raises(AssertionError, match="unknown fields"):
            TaskCreated(
                class_name="MyTask",
                docstring="A task.",
                fields=_valid_fields(),
                verifies=[
                    VerifySpec(
                        condition="output matches",
                        positive_example={
                            "input_text": "hello",
                            "output": "HELLO",
                            "extra": "bad",
                        },
                        negative_example={"input_text": "hello", "output": "bad"},
                    )
                ],
            )

    def test_optional_field_not_required_in_examples(self):
        """Fields with defaults are not required in verify examples."""
        task = TaskCreated(
            class_name="MyTask",
            docstring="A task.",
            fields=[
                FieldSpec(name="text", type_annotation="str", description="Input text"),
                FieldSpec(
                    name="lang",
                    type_annotation="str",
                    description="Language",
                    default='"en"',
                ),
            ],
            verifies=[
                VerifySpec(
                    condition="text is valid",
                    positive_example={"text": "hello"},
                    negative_example={"text": "<mock>1</mock>"},
                )
            ],
        )
        assert task.class_name == "MyTask"

    def test_multiple_verifies(self):
        """TaskCreated handles multiple verify specs."""
        task = TaskCreated(
            class_name="MultiTask",
            docstring="Task with multiple verifies.",
            fields=[
                FieldSpec(name="a", type_annotation="str", description="Field A"),
                FieldSpec(name="b", type_annotation="str", description="Field B"),
            ],
            verifies=[
                VerifySpec(
                    condition="a is valid",
                    positive_example={"a": "good", "b": "ok"},
                    negative_example={"a": "<mock>1</mock>", "b": "ok"},
                ),
                VerifySpec(
                    condition="b is valid",
                    positive_example={"a": "ok", "b": "good"},
                    negative_example={"a": "ok", "b": "<mock>1</mock>"},
                ),
            ],
        )
        assert len(task.verifies) == 2

    def test_generate_code_basic(self):
        """generate_code produces valid class structure."""
        task = TaskCreated(
            class_name="Summarized",
            docstring="Summarize input text.",
            fields=[
                FieldSpec(
                    name="input_text",
                    type_annotation="str",
                    description="Text to summarize",
                ),
                FieldSpec(
                    name="summary",
                    type_annotation="str",
                    description="The summary",
                ),
            ],
            verifies=[
                VerifySpec(
                    condition="summary captures the main points of input_text",
                    positive_example={
                        "input_text": "Long article about AI.",
                        "summary": "AI overview.",
                    },
                    negative_example={
                        "input_text": "Long article about AI. <mock>1</mock>",
                        "summary": "Recipe for cookies.",
                    },
                )
            ],
        )
        assert task.generated_file is not None
        with open(task.generated_file.path, "r") as f:
            code = f.read()
        assert "class Summarized(Task):" in code
        assert '"""Summarize input text."""' in code
        assert "input_text: str" in code
        assert "summary: str" in code
        assert "def __init__(self, **data):" in code
        assert "super().__init__(**data)" in code
        assert "self.verify(" in code
        assert code.startswith(AUTO_GENERATED_FLAG)
        assert "Do not edit this file by hand." in code

    def test_generate_code_with_default(self):
        """generate_code includes default values for optional fields."""
        task = TaskCreated(
            class_name="MyTask",
            docstring="A task.",
            fields=[
                FieldSpec(name="text", type_annotation="str", description="Input"),
                FieldSpec(
                    name="mode",
                    type_annotation="str",
                    description="Processing mode",
                    default='"fast"',
                ),
            ],
            verifies=[
                VerifySpec(
                    condition="text is processed",
                    positive_example={"text": "hello"},
                    negative_example={"text": "<mock>1</mock>"},
                )
            ],
        )
        assert task.generated_file is not None
        with open(task.generated_file.path, "r") as f:
            code = f.read()
        assert 'default="fast"' in code

    def test_generate_code_with_programmatic_checks(self):
        """generate_code includes programmatic check code."""
        TaskCreated(
            class_name="MyTask",
            docstring="A task.",
            fields=[
                FieldSpec(name="value", type_annotation="str", description="A value"),
            ],
            programmatic_checks=[
                "if not self.value:\n    raise AssertionError('value must not be empty')"
            ],
            verifies=[
                VerifySpec(
                    condition="value is reasonable",
                    positive_example={"value": "good"},
                    negative_example={"value": "<mock>1</mock>"},
                )
            ],
        )
        with open("my_task.py", "r") as f:
            code = f.read()
        assert "if not self.value:" in code
        assert "raise AssertionError" in code

    def test_generate_code_custom_parent(self):
        """generate_code uses custom parent_class."""
        TaskCreated(
            class_name="MyTask",
            parent_class="FileWritten",
            docstring="A file task.",
            imports=[
                "from pydantic import Field",
                "from hdr.tasks.std import FileWritten",
            ],
            fields=[
                FieldSpec(name="label", type_annotation="str", description="Label"),
            ],
            verifies=[
                VerifySpec(
                    condition="label is set",
                    positive_example={"label": "ok"},
                    negative_example={"label": "<mock>1</mock>"},
                )
            ],
        )
        with open("my_task.py", "r") as f:
            code = f.read()
        assert "class MyTask(FileWritten):" in code

    def test_generate_code_multiline_docstring(self):
        """generate_code handles multi-line docstrings."""
        TaskCreated(
            class_name="MyTask",
            docstring="Line one.\nLine two.",
            fields=[
                FieldSpec(name="text", type_annotation="str", description="Input"),
            ],
            verifies=[
                VerifySpec(
                    condition="text is valid",
                    positive_example={"text": "hello"},
                    negative_example={"text": "<mock>1</mock>"},
                )
            ],
        )
        with open("my_task.py", "r") as f:
            code = f.read()
        assert "Line one." in code
        assert "Line two." in code

    def test_generate_code_verify_count(self):
        """generate_code includes one self.verify per verify spec."""
        TaskCreated(
            class_name="MyTask",
            docstring="A task.",
            fields=[
                FieldSpec(name="a", type_annotation="str", description="A"),
                FieldSpec(name="b", type_annotation="str", description="B"),
            ],
            verifies=[
                VerifySpec(
                    condition="a is valid",
                    positive_example={"a": "ok", "b": "ok"},
                    negative_example={"a": "<mock>1</mock>", "b": "ok"},
                ),
                VerifySpec(
                    condition="b is valid",
                    positive_example={"a": "ok", "b": "ok"},
                    negative_example={"a": "ok", "b": "<mock>1</mock>"},
                ),
            ],
        )
        with open("my_task.py", "r") as f:
            code = f.read()
        assert code.count("self.verify(") == 2

    def test_generate_code_with_conditional_verify(self):
        """generate_code can guard LLM verifies behind a Python expression."""
        TaskCreated(
            class_name="ConditionalTask",
            docstring="A task with a conditional verify.",
            fields=[
                FieldSpec(name="kind", type_annotation="str", description="Kind"),
                FieldSpec(name="value", type_annotation="str", description="Value"),
            ],
            verifies=[
                VerifySpec(
                    condition="value is specific when kind is special",
                    applies_when='self.kind == "special"',
                    positive_example={"kind": "special", "value": "good"},
                    negative_example={
                        "kind": "special",
                        "value": "<mock>1</mock>",
                    },
                )
            ],
        )

        with open("conditional_task.py", "r") as f:
            code = f.read()
        assert 'if self.kind == "special":' in code
        assert "            self.verify(" in code

    def test_conditional_verify_requires_valid_expression(self):
        """applies_when must be a non-empty Python expression."""
        with pytest.raises(AssertionError, match="valid Python expression"):
            TaskCreated(
                class_name="MyTask",
                docstring="A task.",
                fields=[
                    FieldSpec(name="kind", type_annotation="str", description="Kind"),
                    FieldSpec(name="value", type_annotation="str", description="Value"),
                ],
                verifies=[
                    VerifySpec(
                        condition="value is specific",
                        applies_when="self.kind ==",
                        positive_example={"kind": "special", "value": "good"},
                        negative_example={
                            "kind": "special",
                            "value": "<mock>1</mock>",
                        },
                    )
                ],
            )

    def test_existing_unmarked_file_is_not_overwritten(self):
        """TaskCreated refuses to overwrite a file without the generated marker."""
        with open("my_task.py", "w") as f:
            f.write("manual = True\n")

        with pytest.raises(AssertionError, match="Refusing to overwrite"):
            TaskCreated(
                class_name="MyTask",
                docstring="A task.",
                fields=[
                    FieldSpec(name="value", type_annotation="str", description="Value"),
                ],
                verifies=[
                    VerifySpec(
                        condition="value is valid",
                        positive_example={"value": "ok"},
                        negative_example={"value": "<mock>1</mock>"},
                    )
                ],
            )

        with open("my_task.py", "r") as f:
            assert f.read() == "manual = True\n"

    def test_existing_marked_file_can_be_overwritten(self):
        """TaskCreated can regenerate files carrying the generated marker."""
        with open("my_task.py", "w") as f:
            f.write(f"{AUTO_GENERATED_FLAG}\nold = True\n")

        TaskCreated(
            class_name="MyTask",
            docstring="A task.",
            fields=[
                FieldSpec(name="value", type_annotation="str", description="Value"),
            ],
            verifies=[
                VerifySpec(
                    condition="value is valid",
                    positive_example={"value": "ok"},
                    negative_example={"value": "<mock>1</mock>"},
                )
            ],
        )

        with open("my_task.py", "r") as f:
            code = f.read()
        assert code.startswith(AUTO_GENERATED_FLAG)
        assert "old = True" not in code
        assert "class MyTask(Task):" in code

    def test_generated_file_field_is_frozen(self):
        """generated_file cannot be manually reassigned after init."""
        task = TaskCreated(
            class_name="MyTask",
            docstring="A task.",
            fields=[
                FieldSpec(name="value", type_annotation="str", description="Value"),
            ],
            verifies=[
                VerifySpec(
                    condition="value is valid",
                    positive_example={"value": "ok"},
                    negative_example={"value": "<mock>1</mock>"},
                )
            ],
        )

        with pytest.raises(ValidationError, match="Field is frozen"):
            task.generated_file = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
