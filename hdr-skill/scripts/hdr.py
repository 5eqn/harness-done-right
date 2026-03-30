import os
import json
import re
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

def load_config():
    """Load configuration from ~/.hdr/config.json"""
    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
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
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    success: bool = True,
    error: str = None
):
    """Log LLM call to ~/.hdr/llm_logs.jsonl"""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": request_type,
        "model": model,
        "prompt": prompt,
        "response": response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "success": success,
        "error": error
    }
    os.makedirs(HDR_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

@persist
def _llm_assert_call(condition: str, api_key: str, model: str) -> tuple[str, int, int, int]:
    """Internal cached LLM call for llm_assert"""
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": """You are a strict validator. Evaluate if the following condition is true.
First, output your thinking process in <think> tags.
Then, output a score from 1 to 5 (inclusive) indicating how well the condition is satisfied, where 5 means completely satisfied and 1 means completely unsatisfied.
Only output a score of 5 if the condition is 100% true with no exceptions.

Example output format:
<think>
I need to check if "2 + 2 equals 4" is true. 2 plus 2 is indeed 4, so this condition is completely true.
</think>
Score: 5
"""},
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
    config = load_config()
    model = config.get("openrouter_model")

    if not model:
        error_msg = "openrouter_model is not configured. Please configure it in ~/.hdr/config.json before using llm_assert."
        log_llm_call(
            request_type="assert",
            prompt=condition,
            response="",
            model="unknown",
            success=False,
            error=error_msg
        )
        raise EnvironmentError(error_msg)

    # Handle mock mode
    if model == "mock":
        log_llm_call(
            request_type="assert",
            prompt=condition,
            response="Mock pass",
            model="mock",
            success=True
        )
        return

    # Check for API key
    api_key = config.get("openrouter_api_key")
    if not api_key:
        error_msg = "openrouter_api_key is not configured. Please configure it in ~/.hdr/config.json before using llm_assert."
        log_llm_call(
            request_type="assert",
            prompt=condition,
            response="",
            model=model,
            success=False,
            error=error_msg
        )
        raise EnvironmentError(error_msg)

    try:
        result, input_tokens, output_tokens, total_tokens = _llm_assert_call(condition, api_key, model)

        # Parse result
        think_match = re.search(r'<think>(.*?)</think>', result, re.DOTALL)
        score_match = re.search(r'Score:\s*(\d+)', result)

        thinking = think_match.group(1).strip() if think_match else "No thinking provided"
        score = int(score_match.group(1)) if score_match else 0

        success = score == 5

        # Log the call
        log_llm_call(
            request_type="assert",
            prompt=condition,
            response=result,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            success=success
        )

        if not success:
            raise AssertionError(f"LLM assertion failed with score {score}/5.\nThinking: {thinking}\nCondition: {condition}")
    except Exception as e:
        log_llm_call(
            request_type="assert",
            prompt=condition,
            response="",
            model=model,
            success=False,
            error=str(e)
        )
        raise

# Export all public functions
__all__ = [
    "llm_assert",
    "BaseModel",
    "load_config",
    "save_config",
    "HDR_DIR",
    "LOG_FILE",
    "CONFIG_FILE"
]
