"""
Task specification - THIS FILE IS IMMUTABLE ONCE AGREED
Defines the formal requirements for the task
"""

from hdr import BaseModel, verify, quote
from hdr.tasks.std import File


# Define subtask types
class IntroductionSection(BaseModel):
    file: File

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"{quote(self.file)} is a clear introduction explaining what HDR is")
        verify(f"{quote(self.file)} mentions the core benefits of using HDR")


class UsageSection(BaseModel):
    file: File
    code_examples: list[File]

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"{quote(self.file)} clearly explains how to use HDR")
        for i, example in enumerate(self.code_examples):
            verify(f"Code example {i+1} in {quote(example)} is correct and runnable")


class Documentation(BaseModel):
    title: str
    introduction: IntroductionSection
    usage: UsageSection

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"{quote(self.title)} is clear and descriptive")
        verify(f"{quote(self.introduction)} properly leads into the usage section")
        verify(f"{quote(self)} as a whole is easy to understand for new users")
