"""
Task specification - THIS FILE IS IMMUTABLE ONCE AGREED
Defines the formal requirements for the task
"""
from hdr import BaseModel, verify, quote

# Define subtask types
class IntroductionSection(BaseModel):
    content: str

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"{quote(self.content)} is a clear introduction explaining what HDR is")
        verify(f"{quote(self.content)} mentions the core benefits of using HDR")

class UsageSection(BaseModel):
    content: str
    code_examples: list[str]

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"{quote(self.content)} clearly explains how to use HDR")
        verify(f"All code examples in {quote(self.code_examples)} are correct and runnable")

class Documentation(BaseModel):
    title: str
    introduction: IntroductionSection
    usage: UsageSection

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"{quote(self.title)} is clear and descriptive")
        verify(f"{quote(self.introduction)} properly leads into the usage section")
        verify(f"{quote(self)} as a whole is easy to understand for new users")
