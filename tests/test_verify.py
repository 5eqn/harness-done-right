"""
Tests for HDR verify functionality.
"""

import pytest
from hdr import Task
from pydantic import Field


def test_task_verify_method():
    """Test Task.verify method with automatic context injection"""

    # Create a test Task subclass
    class TestTask(Task):
        value: int = Field(description="Test value")
        name: str = Field(description="Test name")

    # Verify runs without error in mock mode (default score 5)
    task = TestTask(value=42, name="test")
    task.verify("value is 42")
    task.verify("name is 'test'")
    task.verify("value is a positive integer")


def test_verify_mock_score_parsing():
    """Test that verify correctly parses <mock>N</mock> pattern in pytest mode"""

    class TestTask(Task):
        value: int = Field(description="Test value")

    task = TestTask(value=42)

    # Default mock score is 5, should pass with expected_score=5
    task.verify("default mock score")

    # Explicit mock score 5 should pass
    task.verify("explicit 5 <mock>5</mock>")

    # Mock score 4 should fail with expected_score=5
    with pytest.raises(AssertionError, match="Mock verification failed with score 4"):
        task.verify("fail with 4 <mock>4</mock>")

    # Mock score 3 should pass if expected_score=3
    task.verify("pass with 3 <mock>3</mock>", expected_score=3)

    # Mock score 1 should fail with expected_score=2
    with pytest.raises(AssertionError, match="Mock verification failed with score 1"):
        task.verify("fail with 1 <mock>1</mock>", expected_score=2)


if __name__ == "__main__":
    pytest.main([__file__])
