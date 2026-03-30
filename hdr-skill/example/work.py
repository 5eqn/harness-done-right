"""
Task implementation - this file builds the final task instance
Import all task types from task.py (which should not be modified)
"""
from task import *

print("Building documentation components...")

# Build dependency instances first
intro = IntroductionSection(
    content="HDR (Harness Done Right) is a Python library that helps you formalize tasks as code and validate their completion using LLM assertions. It provides automatic type checking, caching of LLM calls, and a stateless workflow that makes task execution predictable and reproducible."
)

usage = UsageSection(
    content="To use HDR, first define your task classes as subclasses of BaseModel, adding llm_assert calls in the constructor to validate completion. Then construct instances of your classes directly - if instantiation succeeds, your task is complete. Configure your OpenRouter API key and model in the settings before running your code.",
    code_examples=[
        "class MyTask(BaseModel):\n    field: str\n    def __init__(self, **data):\n        super().__init__(**data)\n        llm_assert(f\"{quote(self.field)} is valid\")",
        "instance = MyTask(field=\"test value\")"
    ]
)

# Build final target instance
doc = Documentation(
    title="HDR Documentation",
    introduction=intro,
    usage=usage
)

print("\n✅ Task completed successfully!")
print(f"Generated documentation for: {doc.title}")
print(f"Introduction length: {len(doc.introduction.content)} characters")
print(f"Usage section has {len(doc.usage.code_examples)} code examples")
