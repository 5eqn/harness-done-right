"""
Task specification - THIS FILE IS IMMUTABLE ONCE AGREED
Defines the formal requirements for the task
"""

from hdr.tasks.std import File, PythonWorkspace, Context, Task, Concept
from pydantic import Field


class IntroSection(Task):
    """
    Introduction section that explains what the subject is and why it matters.
    """

    context: Context = Field(description="Authoritative definition of the subject being introduced")
    title: str = Field(description="Title of the introduction section")
    file: File = Field(description="Markdown file (.md) containing the introduction text")

    def __init__(self, **data):
        super().__init__(**data)

        self.verify(
            "The file does not contradict any statement in context."
        )
        self.verify(
            "The file contains both (a) a description of what the subject is, and (b) at least one statement of why it is useful or when to use it."
        )
        self.verify(
            "The file only discusses topics within the scope indicated by title; it does not digress into unrelated subjects."
        )
        self.verify(
            "The file contains no time-sensitive terms (e.g., 'currently', 'recently', 'as of now') without specifying an exact version or date."
        )
        self.verify(
            "Every technical term used in file is either defined in context or explained within file itself."
        )


class UsageSection(Task):
    """
    Usage section that explains how to use a concept with runnable examples.
    """

    context: Context = Field(description="Authoritative definition of the concept being documented")
    concept: str = Field(description="Name of the concept being documented")
    file: File = Field(description="Markdown file (.md) containing the usage explanation")
    code_examples: PythonWorkspace = Field(
        description="Python workspace containing runnable code examples"
    )

    def __init__(self, **data):
        super().__init__(**data)

        self.verify(
            "The file does not contradict any statement in context."
        )
        self.verify(
            "Every code snippet in file has a corresponding file in code_examples, and every file in code_examples is referenced in file."
        )
        self.verify(
            "The file only explains usage of concept; it does not introduce or explain usage of unrelated concepts."
        )
        self.verify(
            "The file contains no time-sensitive terms (e.g., 'currently', 'recently', 'as of now') without specifying an exact version or date."
        )
        self.verify(
            "Every technical term used in file is either defined in context or explained within file itself."
        )


class Documentation(Task):
    """
    Complete documentation combining introduction and usage sections.
    """

    intro: IntroSection = Field(description="Introduction section explaining what and why")
    usage: UsageSection = Field(description="Usage section explaining how with examples")