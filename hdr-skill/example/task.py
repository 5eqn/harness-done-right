"""
Task specification - THIS FILE IS IMMUTABLE ONCE AGREED
Defines the formal requirements for the task
"""
from hdr import BaseModel, llm_assert, quote

# Define subtask types
class IntroductionSection(BaseModel):
    content: str

    def __init__(self, **data):
        super().__init__(**data)
        llm_assert(f"{quote(self.content)} is a clear introduction explaining what HDR is")
        llm_assert(f"{quote(self.content)} mentions the core benefits of using HDR")

class UsageSection(BaseModel):
    content: str
    code_examples: list[str]

    def __init__(self, **data):
        super().__init__(**data)
        llm_assert(f"{quote(self.content)} clearly explains how to use HDR")
        llm_assert(f"All code examples in {quote(self.code_examples)} are correct and runnable")
        llm_assert(f"{quote(self.content)} mentions both mock mode and real LLM mode usage")

class Documentation(BaseModel):
    title: str
    introduction: IntroductionSection
    usage: UsageSection

    def __init__(self, **data):
        super().__init__(**data)
        llm_assert(f"{quote(self.title)} is clear and descriptive")
        llm_assert(f"{quote(self.introduction)} properly leads into the usage section")
        llm_assert(f"{quote(self)} as a whole is easy to understand for new users")
