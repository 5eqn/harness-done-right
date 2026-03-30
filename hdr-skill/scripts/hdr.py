import os
from typing import Any
import openai
from pydantic import BaseModel, ValidationError
from locache import persist

# Mock LLM mode
_mock_mode = False
_mock_responses: list[bool] = []

def _check_openrouter_config():
    _openrouter_api_key = None
    _openrouter_model = None
    if not _mock_mode:
        _openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        _openrouter_model = os.getenv("OPENROUTER_MODEL")

        if not _openrouter_api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Please configure it to use HDR with real LLM validation."
            )
        if not _openrouter_model:
            raise EnvironmentError(
                "OPENROUTER_MODEL environment variable is not set. "
                "Please specify which model to use (e.g. 'anthropic/claude-3-opus')."
            )
    return _openrouter_api_key, _openrouter_model

class mock_llm:
    @staticmethod
    def enable():
        """Enable mock LLM mode for testing"""
        global _mock_mode
        _mock_mode = True

    @staticmethod
    def disable():
        """Disable mock LLM mode"""
        global _mock_mode
        _mock_mode = False
        _mock_responses.clear()

    @staticmethod
    def add_response(response: bool):
        """Add a mock response for the next llm_assert/llm_check call"""
        _mock_responses.append(response)

@persist
def llm_assert(condition: str) -> None:
    """
    Validate a condition using LLM. Throws an error with explanation if validation fails.
    Results are cached to avoid duplicate LLM calls.
    """
    if _mock_mode:
        if _mock_responses:
            result = _mock_responses.pop(0)
            if not result:
                raise AssertionError(f"Mock LLM assertion failed: {condition}")
            return
        # Default to passing if no mock responses set
        return

    _openrouter_api_key, _openrouter_model = _check_openrouter_config()

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_openrouter_api_key,
    )

    response = client.chat.completions.create(
        model=_openrouter_model,
        messages=[
            {"role": "system", "content": "You are a strict validator. Evaluate if the following condition is true. Respond only with 'PASS' if it is true, or 'FAIL: [explanation]' if it is false."},
            {"role": "user", "content": condition}
        ]
    )

    result = response.choices[0].message.content.strip()
    if result.startswith("FAIL"):
        raise AssertionError(f"LLM assertion failed: {result[5:].strip()}")

@persist
def llm_check(predicate: str, value: Any) -> bool:
    """
    Run a predicate check using LLM and return a boolean result.
    Results are cached to avoid duplicate LLM calls.
    """
    if _mock_mode:
        if _mock_responses:
            return _mock_responses.pop(0)
        return True

    _openrouter_api_key, _openrouter_model = _check_openrouter_config()

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_openrouter_api_key,
    )

    response = client.chat.completions.create(
        model=_openrouter_model,
        messages=[
            {"role": "system", "content": "Evaluate if the predicate applies to the given value. Respond only with 'YES' or 'NO'."},
            {"role": "user", "content": f"Predicate: {predicate}\nValue: {repr(value)}"}
        ]
    )

    result = response.choices[0].message.content.strip()
    return result == "YES"

# Export all public functions
__all__ = [
    "llm_assert",
    "llm_check",
    "mock_llm",
    "BaseModel"
]
