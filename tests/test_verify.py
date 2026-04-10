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


def test_verify_creates_config_template_when_missing(tmp_path, monkeypatch):
    """verify() should create ~/.hdr/config.yaml and ask the user to fill it in."""

    class TestTask(Task):
        value: int = Field(description="Test value")

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Task, "_CACHE_DIR", str(tmp_path / "cache"))

    task = TestTask(value=42)

    with pytest.raises(EnvironmentError, match="HDR config created at"):
        task.verify("value is 42")

    config_path = tmp_path / ".hdr" / "config.yaml"
    assert config_path.exists()
    content = config_path.read_text()
    assert 'anthropic_auth_token: ""' in content
    assert 'anthropic_model: "claude-4.6-sonnet"' in content
    assert 'anthropic_base_url: "https://api.anthropic.com"' in content


def test_verify_rejects_blank_token_in_config(tmp_path, monkeypatch):
    """verify() should fail clearly when the config file exists but token is blank."""

    class TestTask(Task):
        value: int = Field(description="Test value")

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Task, "_CACHE_DIR", str(tmp_path / "cache"))

    config_dir = tmp_path / ".hdr"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        'anthropic_auth_token: ""\n'
        'anthropic_model: "claude-4.6-sonnet"\n'
        'anthropic_base_url: "https://api.anthropic.com"\n'
    )

    task = TestTask(value=42)

    with pytest.raises(EnvironmentError, match="anthropic_auth_token is empty"):
        task.verify("value is 42")


def test_verify_uses_config_values(tmp_path, monkeypatch):
    """verify() should read API settings from ~/.hdr/config.yaml."""

    class TestTask(Task):
        value: int = Field(description="Test value")

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Task, "_CACHE_DIR", str(tmp_path / "cache"))

    config_dir = tmp_path / ".hdr"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        'anthropic_auth_token: "test-token"\n'
        'anthropic_model: "test-model"\n'
        'anthropic_base_url: "https://example.com"\n'
    )

    captured: dict[str, str] = {}

    def fake_call_llm_with_retry(
        self,
        full_condition: str,
        api_key: str,
        base_url: str,
        model: str,
        max_retries: int = 10,
        verbose: bool = False,
    ) -> tuple[str, int]:
        captured["full_condition"] = full_condition
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        captured["model"] = model
        return "Looks correct.", 5

    monkeypatch.setattr(Task, "_call_llm_with_retry", fake_call_llm_with_retry)

    task = TestTask(value=42)
    task.verify("value is 42")

    assert captured["api_key"] == "test-token"
    assert captured["base_url"] == "https://example.com"
    assert captured["model"] == "test-model"
    assert "This condition holds true: value is 42" in captured["full_condition"]


if __name__ == "__main__":
    pytest.main([__file__])
