"""
Tests for HDR verify functionality.
"""

import pytest
from hdr import quote, BaseModel, Task
from hdr.tasks.std import File
from pydantic import ValidationError, Field


# Test classes


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
    # Test primitive types
    assert quote("test string") == "'test string'"
    assert quote(42) == "42"
    assert quote(3.14) == "3.14"
    assert quote(True) == "True"

    # Test list
    quoted_list = quote([1, 2, 3])
    assert "1," in quoted_list
    assert "2," in quoted_list
    assert "3," in quoted_list

    # Test dict
    test_dict = {"key": "value", "number": 42}
    quoted_dict = quote(test_dict)
    assert "'key': 'value'" in quoted_dict
    assert "'number': 42" in quoted_dict

    # Test BaseModel (should be pretty printed with class name)
    nested = NestedItem(value=42)
    quoted_model = quote(nested)
    assert "NestedItem(" in quoted_model
    assert "value = 42" in quoted_model

    # Test nested BaseModel
    parent = ParentTask(name="test", item=nested)
    quoted_parent = quote(parent)
    assert "ParentTask(" in quoted_parent
    assert "name = 'test'" in quoted_parent
    assert "item = " in quoted_parent
    assert "NestedItem(" in quoted_parent
    assert "value = 42" in quoted_parent

    # Test that quote works with BaseModel instances
    task = StringTask(name="test task")
    quoted_task = quote(task)
    assert "StringTask(" in quoted_task
    assert "name = 'test task'" in quoted_task


def test_task_verify_method():
    """Test Task.verify method with automatic context injection"""

    # Create a test Task subclass
    class TestTask(Task):
        value: int = Field(description="Test value")
        name: str = Field(description="Test name")

    # Verify runs without error in mock mode
    task = TestTask(value=42, name="test")
    task.verify("value is 42")
    task.verify("name is 'test'")
    task.verify("value is a positive integer")


def test_quote_includes_field_descriptions():
    """Test that quote function includes field descriptions in output"""

    class TestTask(Task):
        value: int = Field(description="Test integer value")
        name: str = Field(description="Name of the test object")
        is_active: bool = Field(description="Whether the task is active")

    task = TestTask(value=42, name="test task", is_active=True)
    quoted = quote(task)

    # Check that descriptions are included
    assert "# Test integer value" in quoted
    assert "# Name of the test object" in quoted
    assert "# Whether the task is active" in quoted

    # Check that field values are present
    assert "value = 42" in quoted
    assert "name = 'test task'" in quoted
    assert "is_active = True" in quoted


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
