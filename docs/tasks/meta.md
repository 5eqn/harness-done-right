# Meta Tasks

Meta-task utilities for working with HDR task definitions themselves. These live in `hdr.tasks.meta`.

## TaskCreated

Meta-task for creating well-formed HDR tasks. Validates task definitions at construction time and generates complete task class code with examples embedded in verify conditions.

**Key Validation Features:**
- Validates class names and field names are valid Python identifiers
- Ensures all fields have proper descriptions
- Validates that verify examples include all required fields and no extra fields
- LLM-verifies that positive examples pass the condition (score 5) and negative examples fail (score 1)
- Generates production-ready task class code with embedded examples for clarity

**Fields:**
- `class_name: str` — Name of the task class to create
- `parent_class: str = "Task"` — Parent class to inherit from
- `docstring: str` — Docstring for the task class
- `fields: list[FieldSpec]` — List of field specifications (name, type annotation, description, optional default)
- `programmatic_checks: list[str] = []` — Python code snippets for programmatic validations executed before LLM verifies
- `verifies: list[VerifySpec]` — List of verify specifications with positive and negative examples

**Methods:**
- `generate_code() -> str` — Generates complete Python code for the task class, including all validations and verifies with embedded examples

**Example Usage:**
```python
from hdr import TaskCreated

# Define a new task
task_spec = TaskCreated(
    class_name="SentimentAnalysisCompleted",
    docstring="Validates that sentiment analysis has been performed on text.",
    fields=[
        {
            "name": "input_text",
            "type_annotation": "str",
            "description": "Original text to analyze"
        },
        {
            "name": "sentiment",
            "type_annotation": "str",
            "description": "Sentiment classification: 'positive', 'negative', or 'neutral'"
        }
    ],
    programmatic_checks=[
        "if self.sentiment not in ['positive', 'negative', 'neutral']:\n    raise AssertionError(f'Invalid sentiment: {self.sentiment}')"
    ],
    verifies=[
        {
            "condition": "The sentiment correctly reflects the tone of the input text.",
            "positive_example": {
                "input_text": "I love this product! It works perfectly.",
                "sentiment": "positive"
            },
            "negative_example": {
                "input_text": "I love this product! It works perfectly.",
                "sentiment": "negative"
            }
        }
    ]
)

# Generate the task code
print(task_spec.generate_code())
```

The generated code will include all validations and embedded examples in verify conditions, making it self-documenting and robust.
