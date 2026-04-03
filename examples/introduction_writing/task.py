"""
Task specification - THIS FILE IS IMMUTABLE ONCE AGREED
Defines the formal requirements for the task
"""

from hdr import BaseModel, verify, quote
from hdr.tasks.std import File, PythonWorkspace, Context


# Define subtask types
class UsageSection(BaseModel):
    context: Context
    concept: str
    file: File
    code_examples: PythonWorkspace

    def __init__(self, **data):
        super().__init__(**data)

        ctx = f"[Context] Under parent_context={quote(self.context)}, we're writing a usage section for concept={quote(self.concept)}, containing a usage_summary={quote(self.file)} facing human readers, along with some code_examples={quote(self.code_examples)} [Verify]"

        verify(f"{ctx} The ...")


class Documentation(BaseModel):
    title: str
    usage: UsageSection

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"{quote(self.title)} is clear and descriptive")
        verify(f"{quote(self)} as a whole is easy to understand for new users")
