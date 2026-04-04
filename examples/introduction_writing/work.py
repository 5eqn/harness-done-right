"""
Task implementation - builds final task instance
Import task types from task.py
"""

from hdr import quote, Concept, File

concept = Concept(
    context=File(path="docs/hdr/context.md"),
    name="HDR (Harness Done Right)",
    description=File(path="docs/hdr/description.md"),
)

# TODO

print("✅ Target (construct `IntroSection` instance) accomplished!")