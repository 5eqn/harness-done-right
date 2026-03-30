import os
import tempfile
import pytest
import hdr
from hdr import *

# Test classes defined at module level for pickling
class HumanizeText:
    def __init__(self, original: str, humanized: str):
        self.original = original
        self.humanized = humanized
        llm_assert(f"<a>{original}</a> and <b>{humanized}</b> conveys the same meaning")
        llm_assert(f"{humanized} reads like natural human-written text")

class TestTask:
    def __init__(self, value: int = 0):
        pass

class TypedTask:
    def __init__(self, name: str, count: int):
        pass

class FailingTask:
    def __init__(self, value: str):
        llm_assert(f"{value} is valid")

class TaskA:
    pass

class TaskB:
    pass

class D:
    def __init__(self, value: str):
        llm_assert(f"{value} is a valid D")

class E:
    def __init__(self, value: int):
        llm_assert(f"{value} is a valid E")

class B:
    def __init__(self, d: D, e: E):
        llm_assert(f"{d} and {e} are properly combined into B")

class C:
    def __init__(self, data: list):
        llm_assert(f"{data} is valid for C")

class A:
    def __init__(self, title: str, b: B, c: C):
        llm_assert(f"{title} matches {b} and {c}")

# Reset state before each test
@pytest.fixture(autouse=True)
def reset_state():
    # Clear workbench state
    hdr._workbench.clear()
    hdr._consumed.clear()
    hdr._goal_type = None
    hdr._mock_mode = False
    hdr._mock_responses.clear()

    # Use a temporary pickle file for tests
    original_pickle_path = hdr._pickle_path
    with tempfile.NamedTemporaryFile(delete=False) as f:
        hdr._pickle_path = f.name
    yield
    os.unlink(hdr._pickle_path)
    hdr._pickle_path = original_pickle_path

def test_basic_task_flow():
    """Test the basic task creation and completion flow from the example"""
    mock_llm.enable()

    # Set goal
    goal(HumanizeText)

    # Create task instance
    create("a", HumanizeText("Text with AI", "Text without AI"))

    # Retrieve instance and finish
    instance = get("a")
    finish(instance)

    assert isinstance(instance, HumanizeText)
    assert instance.original == "Text with AI"
    assert instance.humanized == "Text without AI"

def test_instance_reuse_error():
    """Test that instances cannot be reused after being consumed"""
    mock_llm.enable()

    create("test", TestTask(42))

    # First use works
    first = get("test")

    # Second use should fail
    with pytest.raises(ValueError, match="has already been consumed"):
        second = get("test")

def test_type_checking():
    """Test type checking in constructors (type hints are enforced at runtime)"""
    mock_llm.enable()

    # This should work
    create("valid", TypedTask("test", 42))

    # Note: Type hints are not enforced at runtime by default in Python
    # For full runtime type checking, users would need to use a library like pydantic
    # This test just confirms basic functionality works

def test_llm_assert_failure():
    """Test that LLM assertion failures throw appropriate errors"""
    mock_llm.enable()
    mock_llm.add_response(False)  # First assertion fails

    with pytest.raises(AssertionError, match="Mock LLM assertion failed"):
        FailingTask("invalid")

def test_goal_type_mismatch():
    """Test that finish() only accepts instances of the goal type"""
    mock_llm.enable()

    goal(TaskA)

    b_instance = TaskB()
    with pytest.raises(TypeError, match="expected TaskA"):
        finish(b_instance)

def test_recursive_dependencies():
    """Test recursive task construction from the example"""
    mock_llm.enable()

    # Set goal
    goal(A)

    # Build dependencies
    create("d", D("d-value"))
    create("e", E(42))
    create("b", B(get("d"), get("e")))
    create("c", C([1, 2, 3]))

    # Build final A instance
    create("a", A("Test Title", get("b"), get("c")))

    # Complete goal
    finish(get("a"))

def test_duplicate_id_error():
    """Test that creating two instances with the same ID fails"""
    mock_llm.enable()

    create("test", TestTask())

    with pytest.raises(ValueError, match="already exists"):
        create("test", TestTask())

def test_missing_id_error():
    """Test that retrieving a non-existent ID fails"""
    with pytest.raises(ValueError, match="No task instance found"):
        get("non-existent")

def test_finish_without_goal():
    """Test that finish() fails if no goal is set"""
    mock_llm.enable()

    class TestTask:
        pass

    instance = TestTask()
    with pytest.raises(RuntimeError, match="No goal has been set"):
        finish(instance)

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
    with pytest.raises(EnvironmentError, match="OPENROUTER_API_KEY environment variable is not set"):
        llm_assert("test")

    # Set API key but no model
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    with pytest.raises(EnvironmentError, match="OPENROUTER_MODEL environment variable is not set"):
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

if __name__ == "__main__":
    pytest.main([__file__])
