import os
import json
import time
from datetime import datetime, timezone
from typing import Any
import openai
from pydantic import BaseModel
from locache import persist

# HDR directory configuration
HDR_DIR = os.path.expanduser("~/.hdr")
LOG_FILE = os.path.join(HDR_DIR, "llm_logs.jsonl")
CONFIG_FILE = os.path.join(HDR_DIR, "config.json")

os.makedirs(HDR_DIR, exist_ok=True)

# Mock LLM mode
_mock_mode = False
_mock_responses: list[bool] = []

def load_config():
    """Load configuration from ~/.hdr/config.json, fallback to env vars"""
    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)

    # Env vars take precedence over config file
    config["openrouter_api_key"] = os.getenv("OPENROUTER_API_KEY", config.get("openrouter_api_key"))
    config["openrouter_model"] = os.getenv("OPENROUTER_MODEL", config.get("openrouter_model"))
    return config

def save_config(config):
    """Save configuration to ~/.hdr/config.json"""
    os.makedirs(HDR_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

def log_llm_call(
    request_type: str,
    prompt: str,
    response: str,
    cached: bool = False,
    cached_input_tokens: int = 0,
    new_input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    success: bool = True,
    error: str = None
):
    """Log LLM call to ~/.hdr/llm_logs.jsonl"""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": request_type,
        "prompt": prompt,
        "response": response,
        "cached": cached,
        "cached_input_tokens": cached_input_tokens,
        "new_input_tokens": new_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "success": success,
        "error": error
    }
    os.makedirs(HDR_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def _check_openrouter_config():
    _openrouter_api_key = None
    _openrouter_model = None
    if not _mock_mode:
        config = load_config()
        _openrouter_api_key = config.get("openrouter_api_key")
        _openrouter_model = config.get("openrouter_model")

        if not _openrouter_api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. Please configure it in ~/.hdr/config.json or set the environment variable."
            )
        if not _openrouter_model:
            raise EnvironmentError(
                "OPENROUTER_MODEL is not set. Please configure it in ~/.hdr/config.json or set the environment variable (e.g. 'anthropic/claude-3-opus')."
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
def _llm_assert_call(condition: str, model: str) -> tuple[str, int, int, int]:
    """Internal cached LLM call for llm_assert"""
    _openrouter_api_key, _ = _check_openrouter_config()
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_openrouter_api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict validator. Evaluate if the following condition is true. Respond only with 'PASS' if it is true, or 'FAIL: [explanation]' if it is false."},
            {"role": "user", "content": condition}
        ]
    )

    result = response.choices[0].message.content.strip()
    usage = response.usage
    return (
        result,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens
    )

def llm_assert(condition: str) -> None:
    """
    Validate a condition using LLM. Throws an error with explanation if validation fails.
    Results are cached to avoid duplicate LLM calls.
    """
    if _mock_mode:
        if _mock_responses:
            result = _mock_responses.pop(0)
            log_llm_call(
                request_type="assert",
                prompt=condition,
                response="PASS" if result else "FAIL",
                cached=False,
                success=result
            )
            if not result:
                raise AssertionError(f"Mock LLM assertion failed: {condition}")
            return
        # Default to passing if no mock responses set
        log_llm_call(
            request_type="assert",
            prompt=condition,
            response="PASS",
            cached=False,
            success=True
        )
        return

    config = load_config()
    model = config.get("openrouter_model")

    try:
        # Check if this is a cached call
        cache_hit = False
        # We'll detect cache hit by checking if the call returns faster, but for simplicity
        # let's just track after the call
        result, prompt_tokens, completion_tokens, total_tokens = _llm_assert_call(condition, model)

        # Log the call
        log_llm_call(
            request_type="assert",
            prompt=condition,
            response=result,
            cached=cache_hit,
            new_input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
            success=not result.startswith("FAIL")
        )

        if result.startswith("FAIL"):
            raise AssertionError(f"LLM assertion failed: {result[5:].strip()}")
    except Exception as e:
        log_llm_call(
            request_type="assert",
            prompt=condition,
            response="",
            cached=False,
            success=False,
            error=str(e)
        )
        raise

@persist
def _llm_check_call(predicate: str, value_repr: str, model: str) -> tuple[str, int, int, int]:
    """Internal cached LLM call for llm_check"""
    _openrouter_api_key, _ = _check_openrouter_config()
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_openrouter_api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Evaluate if the predicate applies to the given value. Respond only with 'YES' or 'NO'."},
            {"role": "user", "content": f"Predicate: {predicate}\nValue: {value_repr}"}
        ]
    )

    result = response.choices[0].message.content.strip()
    usage = response.usage
    return (
        result,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens
    )

def llm_check(predicate: str, value: Any) -> bool:
    """
    Run a predicate check using LLM and return a boolean result.
    Results are cached to avoid duplicate LLM calls.
    """
    value_repr = repr(value)
    if _mock_mode:
        if _mock_responses:
            result = _mock_responses.pop(0)
            log_llm_call(
                request_type="check",
                prompt=f"Predicate: {predicate}\nValue: {value_repr}",
                response="YES" if result else "NO",
                cached=False,
                success=True
            )
            return result
        # Default to True if no mock responses set
        log_llm_call(
            request_type="check",
            prompt=f"Predicate: {predicate}\nValue: {value_repr}",
            response="YES",
            cached=False,
            success=True
        )
        return True

    config = load_config()
    model = config.get("openrouter_model")

    try:
        result, prompt_tokens, completion_tokens, total_tokens = _llm_check_call(predicate, value_repr, model)
        success = result == "YES"

        log_llm_call(
            request_type="check",
            prompt=f"Predicate: {predicate}\nValue: {value_repr}",
            response=result,
            cached=False,
            new_input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
            success=success
        )

        return success
    except Exception as e:
        log_llm_call(
            request_type="check",
            prompt=f"Predicate: {predicate}\nValue: {value_repr}",
            response="",
            cached=False,
            success=False,
            error=str(e)
        )
        raise

# Export all public functions
__all__ = [
    "llm_assert",
    "llm_check",
    "mock_llm",
    "BaseModel",
    "load_config",
    "save_config",
    "HDR_DIR",
    "LOG_FILE",
    "CONFIG_FILE"
]
