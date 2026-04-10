from hdr.tasks.meta import TaskCreated, FieldSpec, VerifySpec
from hdr.tasks.coding import MarkdownFileWritten

# Load our example files
context = MarkdownFileWritten(path="context.md")
desc_good = MarkdownFileWritten(path="description_good.md")
desc_bad1 = MarkdownFileWritten(path="description_bad1.md")  # Violates condition 1
desc_bad2 = MarkdownFileWritten(path="description_bad2.md")  # Violates condition 2
desc_bad3 = MarkdownFileWritten(path="description_bad3.md")  # Violates condition 3
desc_bad4 = MarkdownFileWritten(path="description_bad4.md")  # Violates condition 4
desc_bad5 = MarkdownFileWritten(path="description_bad5.md")  # Violates condition 5

# Define the ConceptDescribed task specification
concept_described_task = TaskCreated(
    class_name="ConceptDescribed",
    parent_class="Task",
    docstring="Represents a documented concept within a context.",
    fields=[
        FieldSpec(
            name="context",
            type_annotation="MarkdownFileWritten",
            description="File explaining the parent context",
        ),
        FieldSpec(
            name="name", type_annotation="str", description="Name of the concept"
        ),
        FieldSpec(
            name="description",
            type_annotation="MarkdownFileWritten",
            description="File containing the concept description",
        ),
    ],
    verifies=[
        VerifySpec(
            condition="The description is written for readers who understand context but do not yet know name; it neither repeats basics from context nor presumes knowledge of sibling/descendant concepts.",
            positive_example={
                "context": context,
                "name": "HDR Task",
                "description": desc_good,
            },
            negative_example={
                "context": context,
                "name": "HDR Task",
                "description": desc_bad1,
            },
        ),
        VerifySpec(
            condition="The concept name represents exactly one atomic idea that cannot be meaningfully split into two independent concepts.",
            positive_example={
                "context": context,
                "name": "HDR Task",
                "description": desc_good,
            },
            negative_example={
                "context": context,
                "name": "HDR Task and Verification",
                "description": desc_bad2,
            },
        ),
        VerifySpec(
            condition="The description contains no time-sensitive terms (e.g., 'currently', 'recently', 'as of now') without specifying an exact version or date.",
            positive_example={
                "context": context,
                "name": "HDR Task",
                "description": desc_good,
            },
            negative_example={
                "context": context,
                "name": "HDR Task",
                "description": desc_bad3,
            },
        ),
        VerifySpec(
            condition="The description identifies (a) a broader category that name belongs to, and (b) a distinguishing property that separates it from other members of that category.",
            positive_example={
                "context": context,
                "name": "HDR Task",
                "description": desc_good,
            },
            negative_example={
                "context": context,
                "name": "HDR Task",
                "description": desc_bad4,
            },
        ),
        VerifySpec(
            condition="A reader familiar with context can determine for any concrete instance whether it belongs to name, with at most minor edge-case ambiguity.",
            positive_example={
                "context": context,
                "name": "HDR Task",
                "description": desc_good,
            },
            negative_example={
                "context": context,
                "name": "HDR Task",
                "description": desc_bad5,
            },
        ),
    ],
)

print("✅ ConceptDescribed task specification created successfully!")
print("\nGenerated task code:")
print("=" * 80)
print(concept_described_task.generate_code())
