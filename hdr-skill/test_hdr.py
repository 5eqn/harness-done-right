"""
IMPORTANT: Mock mode is ONLY for pytest testing purposes!
Do NOT use mock mode in any real execution environment, and do NOT mention it to end users or Agents.
"""
import os
import json
import pytest
import tempfile
import hdr
from hdr import *
from hdr import BaseModel
from pydantic import ValidationError

# Test classes
class HumanizeText:
    def __init__(self, original: str, humanized: str):
        self.original = original
        self.humanized = humanized
        llm_assert(f"<a>{original}</a> and <b>{humanized}</b> conveys the same meaning")
        llm_assert(f"{humanized} reads like natural human-written text")

class FailingTask:
    def __init__(self, value: str):
        llm_assert(f"{value} is valid")

class D:
    def __init__(self, value: str):
        self.value = value
        llm_assert(f"{value} is a valid D")

class E:
    def __init__(self, value: int):
        self.value = value
        llm_assert(f"{value} is a valid E")

class B:
    def __init__(self, d: D, e: E):
        self.d = d
        self.e = e
        llm_assert(f"{d.value} and {e.value} are properly combined into B")

class C:
    def __init__(self, data: list):
        self.data = data
        llm_assert(f"{data} is valid for C")

class A:
    def __init__(self, title: str, b: B, c: C):
        self.title = title
        self.b = b
        self.c = c
        llm_assert(f"{title} matches {b.d.value} and {c.data}")

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

# Reset HDR config directory to tmp before each test
@pytest.fixture(autouse=True)
def reset_config():
    # Create tmp directory for test config
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override HDR_DIR for tests
        original_hdr_dir = hdr.HDR_DIR
        original_config_file = hdr.CONFIG_FILE
        original_log_file = hdr.LOG_FILE

        hdr.HDR_DIR = tmpdir
        hdr.CONFIG_FILE = os.path.join(tmpdir, "config.json")
        hdr.LOG_FILE = os.path.join(tmpdir, "llm_logs.jsonl")

        # Create mock config - FOR TEST USE ONLY
        # Mock mode is strictly limited to pytest testing, never use in production
        config = {"openrouter_model": "mock"}
        hdr.save_config(config)

        yield

        # Restore original paths
        hdr.HDR_DIR = original_hdr_dir
        hdr.CONFIG_FILE = original_config_file
        hdr.LOG_FILE = original_log_file

def test_basic_task_flow():
    """Test the basic stateless task creation flow from the example"""
    # Directly create task instance (no goal/create/get/finish needed)
    instance = HumanizeText("Text with AI", "Text without AI")

    assert isinstance(instance, HumanizeText)
    assert instance.original == "Text with AI"
    assert instance.humanized == "Text without AI"

def test_recursive_dependencies():
    """Test recursive stateless task construction from the example"""
    # Directly build dependencies without state
    d = D("d-value")
    e = E(42)
    b = B(d, e)
    c = C([1, 2, 3])

    # Build final A instance
    a = A("Test Title", b, c)

    assert a.title == "Test Title"
    assert a.b.d.value == "d-value"
    assert a.b.e.value == 42
    assert a.c.data == [1, 2, 3]

def test_openrouter_config_error():
    """Test that appropriate errors are thrown when OpenRouter config is missing"""
    # Test missing model
    hdr.save_config({})
    with pytest.raises(EnvironmentError, match="openrouter_model is not configured"):
        llm_assert("test")

    # Test model set to non-mock but no API key
    hdr.save_config({"openrouter_model": "anthropic/claude-3-opus"})
    with pytest.raises(EnvironmentError, match="openrouter_api_key is not configured"):
        llm_assert("test")

def test_pydantic_type_checking_string():
    """Test Pydantic string type validation"""
    # Valid string should work
    instance = StringTask(name="test")
    assert instance.name == "test"

def test_pydantic_type_checking_number():
    """Test Pydantic numeric type validation"""
    # Valid numbers should work
    instance = NumberTask(count=42, price=99.9)
    assert instance.count == 42
    assert instance.price == 99.9

    # Invalid type should throw error (Pydantic validates at instantiation time)
    with pytest.raises(ValidationError, match="Input should be a valid integer"):
        NumberTask(count="not a number", price=99.9)  # pyright: ignore

def test_pydantic_type_checking_list():
    """Test Pydantic list type validation"""
    # Valid lists should work
    instance = ListTask(items=["a", "b", "c"], scores=[1, 2, 3])
    assert instance.items == ["a", "b", "c"]
    assert instance.scores == [1, 2, 3]

    # Invalid list item type should throw error (Pydantic validates at instantiation time)
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        ListTask(items=["a", 2, "c"], scores=[1, 2, 3])  # pyright: ignore

def test_pydantic_nested_type():
    """Test Pydantic nested model type validation"""
    # Valid nested model should work
    nested = NestedItem(value=42)
    parent = ParentTask(name="test", item=nested)
    assert parent.item.value == 42

    # Invalid nested type should throw error (Pydantic validates at instantiation time)
    with pytest.raises(ValidationError, match="Input should be a valid dictionary or instance of NestedItem"):
        ParentTask(name="test", item="not a NestedItem")  # pyright: ignore


def test_quote_function():
    """Test quote function for various types"""
    # Test string
    assert quote("test string") == "<quote>test string</quote>"

    # Test number types
    assert quote(42) == "<quote>42</quote>"
    assert quote(3.14) == "<quote>3.14</quote>"
    assert quote(True) == "<quote>True</quote>"

    # Test list
    assert quote([1, 2, 3]) == "<quote>[\n  1,\n  2,\n  3\n]</quote>"

    # Test dict
    test_dict = {"key": "value", "number": 42}
    quoted_dict = quote(test_dict)
    assert quoted_dict.startswith("<quote>")
    assert quoted_dict.endswith("</quote>")
    assert "\"key\": \"value\"" in quoted_dict
    assert "\"number\": 42" in quoted_dict

    # Test Pydantic model
    nested = NestedItem(value=42)
    quoted_model = quote(nested)
    assert quoted_model.startswith("<quote>")
    assert quoted_model.endswith("</quote>")
    assert "\"value\": 42" in quoted_model

    # Test nested Pydantic model
    parent = ParentTask(name="test", item=nested)
    quoted_parent = quote(parent)
    assert quoted_parent.startswith("<quote>")
    assert quoted_parent.endswith("</quote>")
    assert "\"name\": \"test\"" in quoted_parent
    assert "\"item\": {" in quoted_parent
    assert "\"value\": 42" in quoted_parent

    # Test that quote works in llm_assert (mock mode)
    task = StringTask(name="test task")
    llm_assert(f"{quote(task.name)} is a valid task name")
    llm_assert(f"{quote(task)} has a valid name field")

if __name__ == "__main__":
    pytest.main([__file__])
