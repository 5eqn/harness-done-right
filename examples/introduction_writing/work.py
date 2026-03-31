"""
Task implementation - builds final task instance
Import task types from task.py (do not modify)
Runs in /tmp/claude/hdr/hdr_no_commit (empty working dir)
"""
from task import *

print("Building documentation components...")

# Build dependency instances first
intro = IntroductionSection(
    content="""HDR is a Python library for formalizing tasks as code, with AI-powered validation, automatic type checking, AI call caching, and predictable stateless workflows."""
)

usage = UsageSection(
    content="""HDR is an easy Python library for task formalization and AI validation—no advanced experience needed. Learn key concepts (BaseModel, verify(), quote()) and follow simple install/use steps to get started quickly.""",
    code_examples=[
        """# HDR end-to-end example: setup + text summarization
# Follow install steps, save as example.py, run with python example.py
from hdr import BaseModel, verify, quote

class SummarizeText(BaseModel):
    original_text: str
    summary: str
    max_words: int = 50

    def __init__(self,** data):
        super().__init__(**data)
        verify(f"{quote(self.summary)} includes key info from {quote(self.original_text)}")
        verify(f"{quote(self.summary)} ≤ {self.max_words} words")

result = SummarizeText(original_text="HDR simplifies task validation with AI.", summary="HDR uses AI for easy task validation.", max_words=20)
print("🎉 Task complete! Summary:", result.summary)""",
        """# HDR user registration validation example
from hdr import BaseModel, verify, quote

class UserRegistration(BaseModel):
    username: str
    email: str
    password: str

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"{quote(self.username)} is 3-20 chars (letters/numbers/_)")
        verify(f"{quote(self.email)} is valid")
        verify(f"{quote(self.password)} is ≥8 chars (upper/lower/number)")

user = UserRegistration(username="john_doe123", email="john@example.com", password="SecurePass123")
print("✅ Valid registration!")"""
    ]
)

# Build final target instance
doc = Documentation(
    title="HDR Getting Started: Task Formalization & AI Validation",
    introduction=intro,
    usage=usage
)

print("✅ Task completed successfully!")
print(f"Generated documentation for: {doc.title}")
print(f"Introduction length: {len(doc.introduction.content)} characters")
print(f"Usage section has {len(doc.usage.code_examples)} code examples")
