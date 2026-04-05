"""
Task implementation - builds final task instance
Import task types from task.py
"""

from hdr import quote, Concept, File

concept = Concept(
    context=File(path="context.md"),
    name="HDR (Harness Done Right)",
    description=File(path="description.md"),
)

# TODO

print("✅ Target (construct `IntroSection` instance) accomplished!")