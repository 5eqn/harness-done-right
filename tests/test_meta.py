"""
Tests for TaskCreated meta-task.

In pytest mode, verify() returns mock score 5 by default, or parses
<mock>N</mock> from the condition string. For TaskCreated to pass verify
in tests, negative examples embed <mock>1</mock> in a string field value
so that neg_condition gets score 1 (matching expected_score=1).
"""

import pytest
import os
from hdr.tasks.meta import TaskCreated, FieldSpec, VerifySpec


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
        code = task.generated_file.content
        assert "class Summarized(Task):" in code
        assert '"""Summarize input text."""' in code
        assert "input_text: str" in code
        assert "summary: str" in code
        assert "def __init__(self, **data):" in code
        assert "super().__init__(**data)" in code
        assert "self.verify(" in code

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
        code = task.generated_file.content
        assert 'default="fast"' in code

    def test_generate_code_with_programmatic_checks(self):
        """generate_code includes programmatic check code."""
        task = TaskCreated(
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
        code = task.generated_file.content
        assert "if not self.value:" in code
        assert "raise AssertionError" in code

    def test_generate_code_custom_parent(self):
        """generate_code uses custom parent_class."""
        task = TaskCreated(
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
        code = task.generated_file.content
        assert "class MyTask(FileWritten):" in code

    def test_generate_code_multiline_docstring(self):
        """generate_code handles multi-line docstrings."""
        task = TaskCreated(
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
        code = task.generated_file.content
        assert "Line one." in code
        assert "Line two." in code

    def test_generate_code_verify_count(self):
        """generate_code includes one self.verify per verify spec."""
        task = TaskCreated(
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
        code = task.generated_file.content
        assert code.count("self.verify(") == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
