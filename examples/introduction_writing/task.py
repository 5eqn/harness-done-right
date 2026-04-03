"""
Task specification - THIS FILE IS IMMUTABLE ONCE AGREED
Defines the formal requirements for the task
"""

from hdr.tasks.std import File, PythonWorkspace, Context, Task
from pydantic import Field


# Define subtask types
class UsageSection(Task):
    context: Context = Field(description="Parent context this usage section belongs to")
    concept: str = Field(description="Name of the concept being documented")
    usage_summary: File = Field(description="File containing the usage summary text")
    code_examples: PythonWorkspace = Field(
        description="Python workspace containing runnable code examples"
    )

    def __init__(self, **data):
        super().__init__(**data)

        self.verify(
            "The usage section clearly explains how to use the concept in practical scenarios"
        )
        self.verify(
            "All code examples are syntactically correct and follow best practices"
        )
        self.verify(
            "The usage summary does not assume prior knowledge of the concept being explained"
        )


class Documentation(Task):
    title: str = Field(description="Title of the documentation")
    usage: UsageSection = Field(description="Usage section of the documentation")

    def __init__(self, **data):
        super().__init__(**data)
        self.verify(
            "The title is clear and accurately describes the documentation content"
        )
        self.verify("The documentation as a whole is easy to understand for new users")
