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
- Automatically creates a valid Python file (snake_case filename from class name) that passes PyRight and Ruff checks
- Refuses to overwrite an existing file unless it starts with the HDR auto-generated marker. Generated files include regeneration guidance and a do-not-edit hint in the header.
- Returns a `PythonFileWritten` instance for the created file in the `generated_file` field

**Fields:**
- `class_name: str` — Name of the task class to create
- `parent_class: str = "Task"` — Parent class to inherit from
- `docstring: str` — Docstring for the task class
- `fields: list[FieldSpec]` — List of field specifications (name, type annotation, description, optional default)
- `imports: list[str] = ["from pydantic import Field", "from hdr.tasks.std import Task"]` — List of import statements to include at the top of the generated file
- `programmatic_checks: list[str] = []` — Python code snippets for programmatic validations executed before LLM verifies
- `verifies: list[VerifySpec]` — List of verify specifications with positive and negative examples
- `generated_file: PythonFileWritten | None` — The generated Python file task instance, available after successful creation (cannot be manually assigned; will always be non-None after successful initialization)

**Methods:**
- `generate_code() -> str` — Generates complete Python code for the task class, including all validations and verifies with embedded examples
- `generate_full_file_content() -> str` — Generates the complete Python file content including imports and class code
- `generate_file_path() -> str` — Generates the snake_case .py filename from the class name

**Example Usage:**
```python
from hdr.tasks.meta import TaskCreated

# Define a new task
task_spec = TaskCreated(
    class_name="SentimentAnalysisCompleted",
    docstring="Validates that sentiment analysis has been performed on text.",
    imports=[
        "from pydantic import Field",
        "from hdr.tasks.std import Task",
    ],
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

# The file is automatically created and validated!
print(f"Created file: {task_spec.generated_file.path}")
print(f"File content:\n{task_spec.generated_file.content}")
```

The generated code will include all validations and embedded examples in verify conditions, making it self-documenting and robust.
