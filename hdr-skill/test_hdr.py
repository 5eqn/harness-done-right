import os
import pytest
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

# Reset state before each test
@pytest.fixture(autouse=True)
def reset_state():
    hdr._mock_mode = False
    hdr._mock_responses.clear()

def test_basic_task_flow():
    """Test the basic stateless task creation flow from the example"""
    mock_llm.enable()

    # Directly create task instance (no goal/create/get/finish needed)
    instance = HumanizeText("Text with AI", "Text without AI")

    assert isinstance(instance, HumanizeText)
    assert instance.original == "Text with AI"
    assert instance.humanized == "Text without AI"

def test_llm_assert_failure():
    """Test that LLM assertion failures throw appropriate errors"""
    mock_llm.enable()
    mock_llm.add_response(False)  # First assertion fails

    with pytest.raises(AssertionError, match="Mock LLM assertion failed"):
        FailingTask("invalid")

def test_recursive_dependencies():
    """Test recursive stateless task construction from the example"""
    mock_llm.enable()

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
    # Save original env vars
    original_key = os.getenv("OPENROUTER_API_KEY")
    original_model = os.getenv("OPENROUTER_MODEL")

    # Clear env vars
    if "OPENROUTER_API_KEY" in os.environ:
        del os.environ["OPENROUTER_API_KEY"]
    if "OPENROUTER_MODEL" in os.environ:
        del os.environ["OPENROUTER_MODEL"]

    # Disable mock mode to trigger real API check
    mock_llm.disable()

    # Test missing API key
    with pytest.raises(EnvironmentError, match="OPENROUTER_API_KEY is not set"):
        llm_assert("test")

    # Set API key but no model
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    with pytest.raises(EnvironmentError, match="OPENROUTER_MODEL is not set"):
        llm_assert("test")

    # Restore original env vars
    if original_key:
        os.environ["OPENROUTER_API_KEY"] = original_key
    elif "OPENROUTER_API_KEY" in os.environ:
        del os.environ["OPENROUTER_API_KEY"]
    if original_model:
        os.environ["OPENROUTER_MODEL"] = original_model
    elif "OPENROUTER_MODEL" in os.environ:
        del os.environ["OPENROUTER_MODEL"]

def test_llm_check():
    """Test llm_check functionality"""
    mock_llm.enable()
    mock_llm.add_response(True)
    mock_llm.add_response(False)

    assert llm_check("is even", 4) == True
    assert llm_check("is even", 5) == False

def test_pydantic_type_checking_string():
    """Test Pydantic string type validation"""
    mock_llm.enable()

    # Valid string should work
    instance = StringTask(name="test")
    assert instance.name == "test"

def test_pydantic_type_checking_number():
    """Test Pydantic numeric type validation"""
    mock_llm.enable()

    # Valid numbers should work
    instance = NumberTask(count=42, price=99.9)
    assert instance.count == 42
    assert instance.price == 99.9

    # Invalid type should throw error (Pydantic validates at instantiation time)
    with pytest.raises(ValidationError, match="Input should be a valid integer"):
        NumberTask(count="not a number", price=99.9)

def test_pydantic_type_checking_list():
    """Test Pydantic list type validation"""
    mock_llm.enable()

    # Valid lists should work
    instance = ListTask(items=["a", "b", "c"], scores=[1, 2, 3])
    assert instance.items == ["a", "b", "c"]
    assert instance.scores == [1, 2, 3]

    # Invalid list item type should throw error (Pydantic validates at instantiation time)
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        ListTask(items=["a", 2, "c"], scores=[1, 2, 3])

def test_pydantic_nested_type():
    """Test Pydantic nested model type validation"""
    mock_llm.enable()

    # Valid nested model should work
    nested = NestedItem(value=42)
    parent = ParentTask(name="test", item=nested)
    assert parent.item.value == 42

    # Invalid nested type should throw error (Pydantic validates at instantiation time)
    with pytest.raises(ValidationError, match="Input should be a valid dictionary or instance of NestedItem"):
        ParentTask(name="test", item="not a NestedItem")

if __name__ == "__main__":
    pytest.main([__file__])
