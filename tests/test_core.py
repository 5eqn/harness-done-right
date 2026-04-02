"""
Tests for HDR verify functionality.

IMPORTANT: Mock mode is ONLY for pytest testing purposes!
Do NOT use mock mode in any real execution environment.
Mock mode should never be mentioned to end users or in production documentation.
"""

import pytest
import hdr
from hdr import verify, quote, BaseModel
from hdr.tasks.std import File
from pydantic import ValidationError

# Enable mock mode for all tests
# WARNING: Mock mode should ONLY be used in pytest unit tests
hdr.set_mock_mode(True)


# Test classes
class HumanizeText:
    def __init__(self, original: str, humanized: str):
        self.original = original
        self.humanized = humanized
        verify(f"<a>{original}</a> and <b>{humanized}</b> conveys the same meaning")
        verify(f"{humanized} reads like natural human-written text")


class FailingTask:
    def __init__(self, value: str):
        verify(f"{value} is valid")


class D:
    def __init__(self, value: str):
        self.value = value
        verify(f"{value} is a valid D")


class E:
    def __init__(self, value: int):
        self.value = value
        verify(f"{value} is a valid E")


class B:
    def __init__(self, d: D, e: E):
        self.d = d
        self.e = e
        verify(f"{d.value} and {e.value} are properly combined into B")


class C:
    def __init__(self, data: list):
        self.data = data
        verify(f"{data} is valid for C")


class A:
    def __init__(self, title: str, b: B, c: C):
        self.title = title
        self.b = b
        self.c = c
        verify(f"{title} matches {b.d.value} and {c.data}")


# Pydantic test classes
class StringTask(BaseModel):
    name: str


class NumberTask(BaseModel):
    count: int
    price: float


class ListTask(BaseModel):
    items: list[str]
    scores: list[int]


class NestedItem(BaseModel):
    value: int


class ParentTask(BaseModel):
    item: NestedItem
    name: str


def test_basic_task_flow():
    """Test the basic stateless task creation flow from the example"""
    instance = HumanizeText("Text with AI", "Text without AI")

    assert isinstance(instance, HumanizeText)
    assert instance.original == "Text with AI"
    assert instance.humanized == "Text without AI"


def test_recursive_dependencies():
    """Test recursive stateless task construction from the example"""
    d = D("d-value")
    e = E(42)
    b = B(d, e)
    c = C([1, 2, 3])

    a = A("Test Title", b, c)

    assert a.title == "Test Title"
    assert a.b.d.value == "d-value"
    assert a.b.e.value == 42
    assert a.c.data == [1, 2, 3]


def test_pydantic_type_checking_string():
    """Test Pydantic string type validation"""
    instance = StringTask(name="test")
    assert instance.name == "test"


def test_pydantic_type_checking_number():
    """Test Pydantic numeric type validation"""
    instance = NumberTask(count=42, price=99.9)
    assert instance.count == 42
    assert instance.price == 99.9

    with pytest.raises(ValidationError, match="Input should be a valid integer"):
        NumberTask(count="not a number", price=99.9)  # type: ignore[reportArgumentType]


def test_pydantic_type_checking_list():
    """Test Pydantic list type validation"""
    instance = ListTask(items=["a", "b", "c"], scores=[1, 2, 3])
    assert instance.items == ["a", "b", "c"]
    assert instance.scores == [1, 2, 3]

    with pytest.raises(ValidationError, match="Input should be a valid string"):
        ListTask(items=["a", 2, "c"], scores=[1, 2, 3])  # type: ignore[reportArgumentType]


def test_pydantic_nested_type():
    """Test Pydantic nested model type validation"""
    nested = NestedItem(value=42)
    parent = ParentTask(name="test", item=nested)
    assert parent.item.value == 42

    with pytest.raises(
        ValidationError,
        match="Input should be a valid dictionary or instance of NestedItem",
    ):
        ParentTask(name="test", item="not a NestedItem")  # type: ignore[reportArgumentType]


def test_quote_function():
    """Test quote function for various types"""
    assert quote("test string") == "<quote>test string</quote>"

    assert quote(42) == "<quote>42</quote>"
    assert quote(3.14) == "<quote>3.14</quote>"
    assert quote(True) == "<quote>True</quote>"

    assert quote([1, 2, 3]) == "<quote>[\n  1,\n  2,\n  3\n]</quote>"

    test_dict = {"key": "value", "number": 42}
    quoted_dict = quote(test_dict)
    assert quoted_dict.startswith("<quote>")
    assert quoted_dict.endswith("</quote>")
    assert '"key": "value"' in quoted_dict
    assert '"number": 42' in quoted_dict

    nested = NestedItem(value=42)
    quoted_model = quote(nested)
    assert quoted_model.startswith("<quote>")
    assert quoted_model.endswith("</quote>")
    assert '"value": 42' in quoted_model

    parent = ParentTask(name="test", item=nested)
    quoted_parent = quote(parent)
    assert quoted_parent.startswith("<quote>")
    assert quoted_parent.endswith("</quote>")
    assert '"name": "test"' in quoted_parent
    assert '"item": {' in quoted_parent
    assert '"value": 42' in quoted_parent

    # Test that quote works in verify (mock mode)
    task = StringTask(name="test task")
    verify(f"{quote(task.name)} is a valid task name")
    verify(f"{quote(task)} has a valid name field")


class TestFile:
    """Unit tests for File task class"""

    def test_file_exists_with_existing_file(self):
        """Test File validation passes when file exists"""
        # This file exists in the repo
        f = File(path="tests/test_core.py")
        assert f.path == "tests/test_core.py"
        assert len(f.content) > 0

    def test_file_exists_with_nonexistent_file(self):
        """Test File validation fails when file does not exist"""
        with pytest.raises(AssertionError, match="does not exist"):
            File(path="nonexistent_file_12345.txt")

    def test_file_with_content(self):
        """Test File accepts explicit content"""
        f = File(path="tests/test_core.py", content="custom content")
        assert f.path == "tests/test_core.py"
        assert f.content == "custom content"


if __name__ == "__main__":
    pytest.main([__file__])
