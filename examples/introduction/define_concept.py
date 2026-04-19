from hdr.contracts.coding import MarkdownFile
from hdr.contracts.meta import FieldSpec, Contract, VerifySpec

target_reader = MarkdownFile(path="target_reader.md")

precision_good = MarkdownFile(path="precision_good.md")
precision_bad = MarkdownFile(path="precision_bad.md")

opening_scene_good = MarkdownFile(path="opening_scene_good.md")
opening_scene_bad = MarkdownFile(path="opening_scene_bad.md")

forward_momentum_good = MarkdownFile(path="forward_momentum_good.md")
forward_momentum_bad = MarkdownFile(path="forward_momentum_bad.md")

should_force_good = MarkdownFile(path="should_force_good.md")
should_force_bad = MarkdownFile(path="should_force_bad.md")

must_force_good = MarkdownFile(path="must_force_good.md")
must_force_bad = MarkdownFile(path="must_force_bad.md")

turning_point_good = MarkdownFile(path="turning_point_good.md")
turning_point_bad = MarkdownFile(path="turning_point_bad.md")

reader_scope_good = MarkdownFile(path="reader_scope_good.md")
reader_scope_bad = MarkdownFile(path="reader_scope_bad.md")

concept_contract = Contract(
    class_name="Concept",
    parent_class="BaseContract",
    docstring=(
        "Represents a concept defined for a specific target reader through a "
        "precise, story-driven explanation."
    ),
    imports=[
        "from pydantic import Field",
        "from hdr.contracts.std import BaseContract",
        "from hdr.contracts.coding import MarkdownFile",
    ],
    fields=[
        FieldSpec(
            name="target_reader",
            type_annotation="MarkdownFile",
            description=(
                "Markdown file describing the shared background the intended "
                "reader is assumed to already know"
            ),
        ),
        FieldSpec(
            name="name",
            type_annotation="str",
            description="Name of the single concept being explained",
        ),
        FieldSpec(
            name="description",
            type_annotation="MarkdownFile",
            description="Markdown file that explains the concept to the target reader",
        ),
    ],
    verifies=[
        VerifySpec(
            condition=(
                "The contract describes exactly one concept. The concept name names "
                "one atomic idea, and the description makes its membership "
                "boundaries clear enough that a target reader can usually tell "
                "what counts as an instance and what does not."
            ),
            positive_example={
                "target_reader": target_reader,
                "name": "Connection Pooling",
                "description": precision_good,
            },
            negative_example={
                "target_reader": target_reader,
                "name": "Connection Pooling and Query Optimization",
                "description": precision_bad,
            },
        ),
        VerifySpec(
            condition=(
                "The first 2-3 sentences place the target reader inside a "
                "concrete situation with specific details such as numbers, "
                "actions, timings, or observable outcomes. They do not open "
                "with a generic definition or broad claim."
            ),
            positive_example={
                "target_reader": target_reader,
                "name": "Connection Pooling",
                "description": opening_scene_good,
            },
            negative_example={
                "target_reader": target_reader,
                "name": "Connection Pooling",
                "description": opening_scene_bad,
            },
        ),
        VerifySpec(
            condition=(
                "After the first 2-3 sentences, a target reader should feel a "
                "specific unresolved tension and want to know what happens next "
                "or how the situation gets resolved. The opening creates forward "
                "momentum rather than merely announcing the topic."
            ),
            positive_example={
                "target_reader": target_reader,
                "name": "Circuit Breaker",
                "description": forward_momentum_good,
            },
            negative_example={
                "target_reader": target_reader,
                "name": "Circuit Breaker",
                "description": forward_momentum_bad,
            },
        ),
        VerifySpec(
            condition=(
                "Before introducing the concept as the solution, the description "
                "shows the 'should' force: the natural default behavior that "
                "feels simple, reasonable, or attractive to the target reader. "
                "It depicts why someone would keep doing the default instead of "
                "only asserting that the default is bad."
            ),
            positive_example={
                "target_reader": target_reader,
                "name": "Dependency Injection",
                "description": should_force_good,
            },
            negative_example={
                "target_reader": target_reader,
                "name": "Dependency Injection",
                "description": should_force_bad,
            },
        ),
        VerifySpec(
            condition=(
                "The description also shows the 'must' force: a concrete moment "
                "where the default behavior undeniably breaks down. This failure "
                "should be more memorable and compelling than the appeal of the "
                "'should' force, so the need for the concept becomes hard to ignore."
            ),
            positive_example={
                "target_reader": target_reader,
                "name": "Dependency Injection",
                "description": must_force_good,
            },
            negative_example={
                "target_reader": target_reader,
                "name": "Dependency Injection",
                "description": must_force_bad,
            },
        ),
        VerifySpec(
            condition=(
                "The concept enters as the turning point that answers the tension "
                "between the 'should' and 'must' forces. The description presents "
                "the concept as the move that changes the situation, not merely as "
                "a feature list or an isolated dictionary definition."
            ),
            positive_example={
                "target_reader": target_reader,
                "name": "Connection Pooling",
                "description": turning_point_good,
            },
            negative_example={
                "target_reader": target_reader,
                "name": "Connection Pooling",
                "description": turning_point_bad,
            },
        ),
        VerifySpec(
            condition=(
                "The description is written for the stated target reader. It may "
                "rely on background explicitly named in target_reader, but it does "
                "not use target_reader to sneak in concept-specific content the "
                "reader is not already assumed to know. The description itself "
                "does the work of introducing the concept."
            ),
            positive_example={
                "target_reader": target_reader,
                "name": "Prompt Injection",
                "description": reader_scope_good,
            },
            negative_example={
                "target_reader": target_reader,
                "name": "Prompt Injection",
                "description": reader_scope_bad,
            },
        ),
    ],
)

print("Concept contract specification created successfully.")
