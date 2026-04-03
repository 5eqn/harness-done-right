# HDR end-to-end example: setup + text summarization
# Follow install steps, save as example.py, run with python example.py
from hdr.tasks.std import Task
from pydantic import Field


class SummarizeText(Task):
    original_text: str = Field(description="Original text to be summarized")
    summary: str = Field(description="Generated summary of the original text")
    max_words: int = Field(
        default=50, description="Maximum allowed word count for the summary"
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.verify("The summary includes all key information from the original text")
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
