"""
Task specification - THIS FILE IS IMMUTABLE ONCE AGREED
Defines the formal requirements for the task
"""

from hdr import BaseModel, verify, quote
from hdr.tasks.std import File, PythonWorkspace, Context


# Define subtask types
class UsageSection(BaseModel):
    context: Context
    usage_summary: File
    code_examples: PythonWorkspace

    def __init__(self, **data):
        super().__init__(**data)
        verify(
            f"[Condition] context={quote(self.context)}, usage_summary={quote(self.usage_summary)} [Verify] The file clearly explains how to use HDR"
        )


class Documentation(BaseModel):
    title: str
    usage: UsageSection

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"{quote(self.title)} is clear and descriptive")
        verify(f"{quote(self)} as a whole is easy to understand for new users")
