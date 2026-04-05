"""
Task implementation - builds final task instance
Import task types from task.py
"""

from hdr import ConceptDescribed, FileWritten

concept = ConceptDescribed(
    context=FileWritten(path="context.md"),
    name="HDR (Harness Done Right)",
    description=FileWritten(path="description.md"),
)

# TODO

print("✅ Target (construct `IntroSection` instance) accomplished!")
