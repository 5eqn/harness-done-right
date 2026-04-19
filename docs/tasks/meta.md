# Meta Contracts

Meta-contract utilities for working with HDR contract definitions themselves. These live in `hdr.contracts.meta`.

## Contract

Meta-contract for creating well-formed HDR contracts. Validates contract definitions at construction time and generates complete contract class code with examples embedded in verify conditions.

**Key Validation Features:**
- Validates class names and field names are valid Python identifiers
- Ensures all fields have proper descriptions
- Validates that verify examples include all required fields and no extra fields
- Validates optional conditional verify guards as Python expressions
- LLM-verifies that positive examples pass the condition (score 5) and negative examples fail (score 1)
- Generates production-ready contract class code with embedded examples for clarity
- Automatically creates a valid Python file (snake_case filename from class name) that passes PyRight and Ruff checks
- Refuses to overwrite an existing file unless it starts with the HDR auto-generated marker. Generated files include regeneration guidance and a do-not-edit hint in the header.
- Returns a `PythonFile` instance for the created file in the `generated_file` field

**Fields:**
- `class_name: str` — Name of the contract class to create
- `parent_class: str = "BaseContract"` — Parent class to inherit from
- `docstring: str` — Docstring for the contract class
- `fields: list[FieldSpec]` — List of field specifications (name, type annotation, description, optional default)
- `imports: list[str] = ["from pydantic import Field", "from hdr.contracts.std import BaseContract"]` — List of import statements to include at the top of the generated file
- `programmatic_checks: list[str] = []` — Python code snippets for programmatic validations executed before LLM verifies
- `verifies: list[VerifySpec]` — List of verify specifications with positive and negative examples. Each `VerifySpec` can include `applies_when: str | None`, a Python expression such as `self.mode == "strict"` that guards the generated LLM verify when the semantic rule only applies to some contract instances.
- `generated_file: PythonFile | None` — The generated Python file instance, available after successful creation (cannot be manually assigned; will always be non-None after successful initialization)

**Methods:**
- `generate_code() -> str` — Generates complete Python code for the contract class, including all validations and verifies with embedded examples
- `generate_full_file_content() -> str` — Generates the complete Python file content including imports and class code
- `generate_file_path() -> str` — Generates the snake_case .py filename from the class name

**Example Usage:**
```python
from hdr.contracts.meta import Contract

# Define a new contract
contract_spec = Contract(
    class_name="SentimentAnalysis",
    docstring="Validates that sentiment analysis has been performed on text.",
    imports=[
        "from pydantic import Field",
        "from hdr.contracts.std import BaseContract",
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
            "applies_when": "self.sentiment in ['positive', 'negative', 'neutral']",
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
print(f"Created file: {contract_spec.generated_file.path}")
print(f"File content:\n{contract_spec.generated_file.content}")
```

The generated code will include all validations and embedded examples in verify conditions, making it self-documenting and robust.
