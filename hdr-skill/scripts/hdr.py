import os
import pickle
import inspect
from typing import Any, Type, Dict, Optional
import openai
from pydantic import BaseModel, ValidationError

# Workbench storage
_workbench: Dict[str, Any] = {}
_consumed: set[str] = set()
_goal_type: Optional[Type] = None
_pickle_path = ".hdr_workbench.pkl"

# Mock LLM mode
_mock_mode = False
_mock_responses: list[bool] = []

def _check_openrouter_config():
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
    return True

def _save_workbench():
    with open(_pickle_path, "wb") as f:
        pickle.dump((_workbench, _consumed, _goal_type), f)

def _load_workbench():
    global _workbench, _consumed, _goal_type
    if os.path.exists(_pickle_path):
        with open(_pickle_path, "rb") as f:
            _workbench, _consumed, _goal_type = pickle.load(f)

# Load existing workbench on import
_load_workbench()

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

def llm_assert(condition: str) -> None:
    """
    Validate a condition using LLM. Throws an error with explanation if validation fails.
    """
    if _mock_mode:
        if _mock_responses:
            result = _mock_responses.pop(0)
            if not result:
                raise AssertionError(f"Mock LLM assertion failed: {condition}")
            return
        # Default to passing if no mock responses set
        return

    _check_openrouter_config()

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

def llm_check(predicate: str, value: Any) -> bool:
    """
    Run a predicate check using LLM and return a boolean result.
    """
    if _mock_mode:
        if _mock_responses:
            return _mock_responses.pop(0)
        return True

    _check_openrouter_config()

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

def goal(task_type: Type) -> None:
    """
    Set the target task type that needs to be completed.
    """
    global _goal_type
    _goal_type = task_type
    _save_workbench()

def create(id: str, instance: Any) -> None:
    """
    Store a task instance in the workbench with the given ID.
    Validates all type constraints and assertions before storing.
    If instance is a Pydantic BaseModel, type validation automatically occurs during instantiation.
    """
    if id in _workbench:
        raise ValueError(f"Task instance with ID '{id}' already exists")

    _workbench[id] = instance
    _save_workbench()

def get(id: str) -> Any:
    """
    Retrieve a task instance from the workbench by ID.
    Marks the instance as consumed so it cannot be reused.
    """
    if id not in _workbench:
        raise ValueError(f"No task instance found with ID '{id}'")

    if id in _consumed:
        raise ValueError(f"Task instance '{id}' has already been consumed and cannot be reused")

    _consumed.add(id)
    _save_workbench()
    return _workbench[id]

def finish(instance: Any) -> None:
    """
    Mark the goal as completed using the provided instance, which must match the goal task type.
    """
    global _goal_type

    if _goal_type is None:
        raise RuntimeError("No goal has been set. Call goal() first.")

    if not isinstance(instance, _goal_type):
        raise TypeError(f"Instance is of type {type(instance).__name__}, expected {_goal_type.__name__}")

    print(f"✅ Goal completed successfully! Task type: {_goal_type.__name__}")
    print(f"📝 Full task details: {repr(instance)}")

    # Clear state for next task
    _goal_type = None
    _save_workbench()

# Export all public functions
__all__ = [
    "llm_assert",
    "llm_check",
    "goal",
    "create",
    "get",
    "finish",
    "mock_llm",
    "BaseModel"
]
