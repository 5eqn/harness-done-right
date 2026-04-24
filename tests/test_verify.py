"""
Tests for HDR verify functionality.
"""

import pytest
import hdr.config as hdr_config
from hdr.contracts.std import BaseContract
from pydantic import Field


def test_contract_verify_method():
    """Test BaseContract.verify method with automatic context injection"""

    # Create a test BaseContract subclass
    class TestContract(BaseContract):
        value: int = Field(description="Test value")
        name: str = Field(description="Test name")

    # Verify runs without error in mock mode (default score 5)
    task = TestContract(value=42, name="test")
    task.llm_verify("value is 42")
    task.llm_verify("name is 'test'")
    task.llm_verify("value is a positive integer")


def test_verify_logs_success_with_score_and_trimmed_condition(capsys):
    """Successful verify() calls log a one-line summary with the actual score."""

    class TestBaseContract(BaseContract):
        value: int = Field(description="Test value")

    task = TestBaseContract(value=42)

    task.llm_verify(
        "value is 42 and the explanation should be trimmed onto one line "
        "because this sentence is intentionally long enough to exceed the log preview limit"
    )

    captured = capsys.readouterr()
    assert captured.out.startswith("[verify] score=5 ")
    assert captured.out.endswith("...\n")
    assert "\n" not in captured.out[:-1]
    assert "expected" not in captured.out
    assert (
        "value is 42 and the explanation should be trimmed onto one l" in captured.out
    )


def test_verify_mock_score_parsing():
    """Test that verify correctly parses <mock>N</mock> pattern in pytest mode"""

    class TestBaseContract(BaseContract):
        value: int = Field(description="Test value")

    task = TestBaseContract(value=42)

    # Default mock score is 5, should pass with expected_score=5
    task.llm_verify("default mock score")

    # Explicit mock score 5 should pass
    task.llm_verify("explicit 5 <mock>5</mock>")

    # Mock score 4 should fail with expected_score=5
    with pytest.raises(AssertionError, match="Mock verification failed with score 4"):
        task.llm_verify("fail with 4 <mock>4</mock>")

    # Mock score 3 should pass if expected_score=3
    task.llm_verify("pass with 3 <mock>3</mock>", expected_score=3)

    # Mock score 1 should fail with expected_score=2
    with pytest.raises(AssertionError, match="Mock verification failed with score 1"):
        task.llm_verify("fail with 1 <mock>1</mock>", expected_score=2)


def test_verify_creates_config_template_when_missing(tmp_path, monkeypatch):
    """verify() should create ~/.hdr/config.yaml and ask the user to fill it in."""

    class TestBaseContract(BaseContract):
        value: int = Field(description="Test value")

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(hdr_config, "CONFIG_PATH", tmp_path / ".hdr" / "config.yaml")

    task = TestBaseContract(value=42)

    with pytest.raises(EnvironmentError, match="HDR config created at"):
        task.llm_verify("value is 42")

    config_path = hdr_config.CONFIG_PATH
    assert config_path.exists()
    content = config_path.read_text()
    assert 'anthropic_auth_token: ""' in content
    assert 'anthropic_model: "claude-4.6-sonnet"' in content
    assert 'anthropic_base_url: "https://api.anthropic.com"' in content
    assert f'verify_cache_dir: "{hdr_config.DEFAULT_VERIFY_CACHE_DIR}"' in content


def test_verify_rejects_blank_token_in_config(tmp_path, monkeypatch):
    """verify() should fail clearly when the config file exists but token is blank."""

    class TestBaseContract(BaseContract):
        value: int = Field(description="Test value")

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(hdr_config, "CONFIG_PATH", tmp_path / ".hdr" / "config.yaml")

    hdr_config.CONFIG_PATH.parent.mkdir()
    hdr_config.CONFIG_PATH.write_text(
        'anthropic_auth_token: ""\n'
        'anthropic_model: "claude-4.6-sonnet"\n'
        'anthropic_base_url: "https://api.anthropic.com"\n'
        f'verify_cache_dir: "{tmp_path / "cache"}"\n'
    )

    task = TestBaseContract(value=42)

    with pytest.raises(EnvironmentError, match="anthropic_auth_token is empty"):
        task.llm_verify("value is 42")


def test_verify_uses_config_values(tmp_path, monkeypatch):
    """verify() should read API settings from ~/.hdr/config.yaml."""

    class TestBaseContract(BaseContract):
        value: int = Field(description="Test value")

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(hdr_config, "CONFIG_PATH", tmp_path / ".hdr" / "config.yaml")

    hdr_config.CONFIG_PATH.parent.mkdir()
    hdr_config.CONFIG_PATH.write_text(
        'anthropic_auth_token: "test-token"\n'
        'anthropic_model: "test-model"\n'
        'anthropic_base_url: "https://example.com"\n'
        f'verify_cache_dir: "{tmp_path / "cache"}"\n'
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

    monkeypatch.setattr(BaseContract, "_call_llm_with_retry", fake_call_llm_with_retry)

    task = TestBaseContract(value=42)
    task.llm_verify("value is 42")

    assert captured["api_key"] == "test-token"
    assert captured["base_url"] == "https://example.com"
    assert captured["model"] == "test-model"
    assert "This condition holds true: value is 42" in captured["full_condition"]


def test_verify_logs_actual_score_on_non_default_success(tmp_path, monkeypatch, capsys):
    """Successful verify() logs the actual score without an expected score suffix."""

    class TestBaseContract(BaseContract):
        value: int = Field(description="Test value")

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(hdr_config, "CONFIG_PATH", tmp_path / ".hdr" / "config.yaml")

    hdr_config.CONFIG_PATH.parent.mkdir()
    hdr_config.CONFIG_PATH.write_text(
        'anthropic_auth_token: "test-token"\n'
        'anthropic_model: "test-model"\n'
        'anthropic_base_url: "https://example.com"\n'
        f'verify_cache_dir: "{tmp_path / "cache"}"\n'
    )

    def fake_call_llm_with_retry(
        self,
        full_condition: str,
        api_key: str,
        base_url: str,
        model: str,
        max_retries: int = 10,
        verbose: bool = False,
    ) -> tuple[str, int]:
        return "Looks mostly correct.", 3

    monkeypatch.setattr(BaseContract, "_call_llm_with_retry", fake_call_llm_with_retry)

    task = TestBaseContract(value=42)
    task.llm_verify("value is acceptable", expected_score=3)

    captured = capsys.readouterr()
    assert captured.out == "[verify] score=3 value is acceptable\n"


if __name__ == "__main__":
    pytest.main([__file__])
