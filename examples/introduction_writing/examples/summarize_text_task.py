# HDR end-to-end example: setup + text summarization
# Follow install steps, save as example.py, run with python example.py
from hdr import BaseModel, verify, quote


class SummarizeText(BaseModel):
    original_text: str
    summary: str
    max_words: int = 50

    def __init__(self, **data):
        super().__init__(**data)
        verify(
            f"{quote(self.summary)} includes key info from {quote(self.original_text)}"
        )
        # Word count check: summary must be ≤ max_words
        assert len(self.summary.split()) <= self.max_words, (
            f"Summary exceeds {self.max_words} words"
        )


result = SummarizeText(
    original_text="HDR simplifies task validation with AI.",
    summary="HDR uses AI for easy task validation.",
    max_words=20,
)
print("🎉 Task complete! Summary:", result.summary)
