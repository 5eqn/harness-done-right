"""
Tests for HDR quote functionality.
"""

import pytest
from hdr.tasks.std import Task, quote
from pydantic import BaseModel, Field


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


if __name__ == "__main__":
    pytest.main([__file__])
