"""
Task implementation - builds final task instance
Import task types from task.py
"""

from task import IntroductionSection, UsageSection, Documentation
from hdr.tasks.std import File, PythonWorkspace

print("Building documentation components...")

# Build dependency instances first
intro = IntroductionSection(
    file=File(path="intro.md")
)

usage = UsageSection(
    file=File(path="usage.md"),
    code_examples=PythonWorkspace(path="examples"),
)

# Build final target instance
doc = Documentation(
    title="HDR Getting Started: Task Formalization & AI Validation",
    introduction=intro,
    usage=usage,
)

print("✅ Task completed successfully!")
