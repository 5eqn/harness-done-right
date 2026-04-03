"""
Task implementation - builds final task instance
Import task types from task.py
"""

from task import UsageSection, Documentation
from hdr.tasks.std import File, PythonWorkspace

print("Building documentation components...")

# Build dependency instances first
usage = UsageSection(
    usage_summary=File(path="usage.md"),
    code_examples=PythonWorkspace(path="examples"),
)

# Build final target instance
doc = Documentation(
    title="HDR Getting Started: Task Formalization & AI Validation",
    usage=usage,
)

print("✅ Task completed successfully!")
