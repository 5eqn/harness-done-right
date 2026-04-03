"""
Task implementation - builds final task instance
Import task types from task.py
"""

from hdr import Concept, File

concept = Concept(
    context=File(path="context.md"),
    name="HDR (Harness Done Right)",
    description=File(path="intro.md"),
)

print("✅ Task completed successfully!")
