from hdr.tasks.coding import MarkdownFile
from hdr.tasks.meta import FieldSpec, Contract, VerifySpec

from move_support import MoveContext, MovePurpose, MoveSnapshot


target_reader = MarkdownFile(path="hdr_target_reader.md")
target_meaning = MarkdownFile(path="hdr_target_meaning.md")

empty_context = MoveContext(
    target_reader=target_reader,
    target_concept="HDR",
    target_meaning=target_meaning,
    prior_moves=[],
)

build_snapshot = MoveSnapshot(
    purpose=MovePurpose.BUILD,
    content=(
        "A careful-looking agent run can still leave rollback as a sentence in "
        "the prompt instead of something the harness can reject."
    ),
    contrast="soft vs enforceable",
    before=(
        "The reader treats a detailed prompt and a long tool-using run as a "
        "reasonable completion boundary."
    ),
    after=(
        "The reader sees that prose-bound requirements stay soft unless the "
        "harness has something enforceable to reject."
    ),
)

context_after_build = MoveContext(
    target_reader=target_reader,
    target_concept="HDR",
    target_meaning=target_meaning,
    prior_moves=[build_snapshot],
)

move_contract = Contract(
    class_name="Move",
    parent_class="BaseContract",
    docstring=(
        "Represents one chosen conceptual move in a language-agnostic outline, "
        "validated as a reader-state transition along one contrast axis."
    ),
    imports=[
        "from pydantic import Field",
        "from hdr.tasks.std import BaseContract",
        "from move_support import MoveContext, MovePurpose",
    ],
    fields=[
        FieldSpec(
            name="context",
            type_annotation="MoveContext",
            description=(
                "Target reader, target concept, target meaning, and prior moves "
                "that make necessity and continuity checkable"
            ),
        ),
        FieldSpec(
            name="purpose",
            type_annotation="MovePurpose",
            description=(
                "Whether the move builds concrete pressure or shifts the reader's "
                "interpretive lens"
            ),
        ),
        FieldSpec(
            name="content",
            type_annotation="str",
            description=(
                "Atomic outline sentence chosen for the move; it is not final prose"
            ),
        ),
        FieldSpec(
            name="contrast",
            type_annotation="str",
            description=(
                "Single projected contrast axis, written as opposing qualities "
                "such as 'soft vs enforceable'"
            ),
        ),
        FieldSpec(
            name="before",
            type_annotation="str",
            description=(
                "Relevant local reader lens before this move, focused only on "
                "the dimension the move changes"
            ),
        ),
        FieldSpec(
            name="after",
            type_annotation="str",
            description=(
                "Relevant local reader lens after this move, focused on the same "
                "dimension as before and contrast"
            ),
        ),
    ],
    programmatic_checks=[
        """
if " vs " not in self.contrast:
    raise AssertionError("contrast must use the form '<quality> vs <opposing quality>'")
contrast_parts = [part.strip() for part in self.contrast.split(" vs ")]
if len(contrast_parts) != 2 or not all(contrast_parts):
    raise AssertionError("contrast must contain exactly two non-empty opposing qualities")
if contrast_parts[0].lower() == contrast_parts[1].lower():
    raise AssertionError("contrast qualities must not be identical")
for field_name in ("content", "before", "after"):
    if not getattr(self, field_name).strip():
        raise AssertionError(f"{field_name} must not be empty")
"""
    ],
    verifies=[
        VerifySpec(
            condition=(
                "The contrast is exactly one projected axis of opposing qualities. "
                "The second side is meaningfully equivalent to 'not the first side' "
                "but with sharper taste language. It must not combine two metrics, "
                "two unrelated properties, object names, or analogy labels."
            ),
            positive_example={
                "context": context_after_build,
                "purpose": MovePurpose.SHIFT,
                "content": (
                    "The failure is not that the agent forgot to care; it is that "
                    "'careful' stayed soft instead of becoming an enforceable "
                    "contract the harness could reject."
                ),
                "contrast": "soft vs enforceable",
                "before": (
                    "The reader sees the missing rollback as a matter of agent "
                    "carefulness."
                ),
                "after": (
                    "The reader sees the missing rollback as a soft constraint "
                    "that needed an enforceable representation."
                ),
            },
            negative_example={
                "context": context_after_build,
                "purpose": MovePurpose.SHIFT,
                "content": (
                    "The run was expensive and still missed the rollback step, so "
                    "HDR should make completion more reliable."
                ),
                "contrast": "high cost vs low guarantee",
                "before": "The reader notices the agent used a lot of tokens.",
                "after": "The reader wants a stronger guarantee.",
            },
        ),
        VerifySpec(
            condition=(
                "The content, before, and after all operate on the same axis named "
                "by contrast. Rich details in content are allowed only when they "
                "clarify or intensify that single axis. The move fails if it "
                "introduces a second independent contrast."
            ),
            positive_example={
                "context": context_after_build,
                "purpose": MovePurpose.SHIFT,
                "content": (
                    "A checklist in prose can remind the agent, but HDR makes the "
                    "same requirement enforceable by putting it into a contract "
                    "that construction can reject."
                ),
                "contrast": "soft vs enforceable",
                "before": (
                    "The reader treats prose checklists as useful reminders for "
                    "agent behavior."
                ),
                "after": (
                    "The reader sees prose checklists as soft reminders and contracts "
                    "as enforceable completion boundaries."
                ),
            },
            negative_example={
                "context": context_after_build,
                "purpose": MovePurpose.SHIFT,
                "content": (
                    "A checklist in prose is fast to write, while a contract "
                    "adds more syntax, more imports, and more setup work."
                ),
                "contrast": "soft vs enforceable",
                "before": "The reader thinks prose is fast to write.",
                "after": "The reader thinks contracts add setup overhead.",
            },
        ),
        VerifySpec(
            condition=(
                "Given context and before, content plausibly causes after without "
                "a hidden reasoning step. Before and after must describe the same "
                "local reader lens, and after must be a meaningful cognition "
                "upgrade rather than a paraphrase, mood change, or unrelated fact."
            ),
            positive_example={
                "context": context_after_build,
                "purpose": MovePurpose.SHIFT,
                "content": (
                    "Once rollback is a field the contract must satisfy, a missing "
                    "rollback stops being a review comment and becomes a "
                    "construction failure."
                ),
                "contrast": "post-hoc vs construction-time",
                "before": (
                    "The reader treats missing rollback as feedback discovered "
                    "after the agent claims completion."
                ),
                "after": (
                    "The reader sees missing rollback as something the contract can "
                    "reject during construction before completion is accepted."
                ),
            },
            negative_example={
                "context": context_after_build,
                "purpose": MovePurpose.SHIFT,
                "content": (
                    "HDR uses Python classes, which many developers already know "
                    "from building ordinary applications."
                ),
                "contrast": "post-hoc vs construction-time",
                "before": "The reader treats missing rollback as post-hoc feedback.",
                "after": "The reader remembers that Python classes are familiar.",
            },
        ),
        VerifySpec(
            condition=(
                "The move is necessary against target_meaning. A strong editor "
                "could not delete content from an excellent introduction without "
                "reducing tension, precision, inevitability, or comprehension. "
                "Vivid but decorative details fail."
            ),
            positive_example={
                "context": empty_context,
                "purpose": MovePurpose.BUILD,
                "content": (
                    "The rollback step can disappear inside a long agent run "
                    "because prose asks for it, but nothing in the harness must "
                    "hold it."
                ),
                "contrast": "requested vs held",
                "before": (
                    "The reader assumes that asking for a rollback step gives the "
                    "agent a durable obligation."
                ),
                "after": (
                    "The reader sees that a prose request is not durable unless "
                    "the harness holds the requirement as data."
                ),
            },
            negative_example={
                "context": empty_context,
                "purpose": MovePurpose.BUILD,
                "content": (
                    "It is 11:47 p.m., the terminal cursor blinks, and the room "
                    "feels quiet while the agent keeps working."
                ),
                "contrast": "quiet vs dramatic",
                "before": "The reader has no image of the scene.",
                "after": "The reader imagines a late-night terminal scene.",
            },
        ),
        VerifySpec(
            condition=(
                "The move uses only concepts available from target_reader, prior "
                "moves, or concepts it explicitly introduces inside content. It "
                "must not rely on unexplained terms, distinctions, metaphors, or "
                "framework-specific meanings."
            ),
            positive_example={
                "context": context_after_build,
                "purpose": MovePurpose.SHIFT,
                "content": (
                    "Calling it a contract matters because this reader already "
                    "knows software contracts: something either satisfies the "
                    "shape and checks, or construction fails."
                ),
                "contrast": "suggestive vs contractual",
                "before": "The reader hears 'careful' as a suggestive instruction.",
                "after": (
                    "The reader connects HDR to the familiar idea of a contractual "
                    "condition that can fail."
                ),
            },
            negative_example={
                "context": empty_context,
                "purpose": MovePurpose.SHIFT,
                "content": (
                    "The move is basically a Curry-Howard proof term over the "
                    "agent's latent meaning vector."
                ),
                "contrast": "informal vs type-theoretic",
                "before": "The reader thinks the agent needs clearer instructions.",
                "after": "The reader understands HDR as a proof-theoretic object.",
            },
        ),
        VerifySpec(
            condition=(
                "When purpose is BUILD, content creates concrete pressure using "
                "a real-looking mechanism, failure tendency, system behavior, or "
                "grounded detail that the target reader can immediately recognize. "
                "It must not be generic abstraction."
            ),
            applies_when="self.purpose == MovePurpose.BUILD",
            positive_example={
                "context": empty_context,
                "purpose": MovePurpose.BUILD,
                "content": (
                    "The agent can edit five files, summarize the migration, and "
                    "still leave rollback as a line nobody has to instantiate."
                ),
                "contrast": "soft vs enforceable",
                "before": (
                    "The reader treats a detailed migration prompt as enough to "
                    "make rollback part of the work."
                ),
                "after": (
                    "The reader sees rollback as soft when it remains prose rather "
                    "than an enforceable object field."
                ),
            },
            negative_example={
                "context": empty_context,
                "purpose": MovePurpose.BUILD,
                "content": (
                    "Agents sometimes miss important requirements, so better "
                    "validation can improve reliability."
                ),
                "contrast": "weak vs strong",
                "before": "The reader knows agents can miss things.",
                "after": "The reader knows validation can help.",
            },
        ),
        VerifySpec(
            condition=(
                "When purpose is BUILD, every vivid detail in content is "
                "load-bearing. Removing the detail would weaken attention, "
                "mechanism understanding, pressure, or preparation for a later "
                "shift. Pure atmosphere fails."
            ),
            applies_when="self.purpose == MovePurpose.BUILD",
            positive_example={
                "context": empty_context,
                "purpose": MovePurpose.BUILD,
                "content": (
                    "A migration plan can name backfill timing, rollback, and "
                    "verification commands in prose, yet none of those nouns are "
                    "objects the harness must inspect before it accepts 'done'."
                ),
                "contrast": "named vs inspected",
                "before": (
                    "The reader treats named requirements in a plan as meaningful "
                    "coverage."
                ),
                "after": (
                    "The reader sees that naming requirements is weaker than "
                    "making them inspectable by the harness."
                ),
            },
            negative_example={
                "context": empty_context,
                "purpose": MovePurpose.BUILD,
                "content": (
                    "The migration plan sits in a long markdown file with neat "
                    "headings, clean bullets, and a timestamp at the top."
                ),
                "contrast": "plain vs polished",
                "before": "The reader imagines an ordinary markdown plan.",
                "after": "The reader imagines a better-formatted markdown plan.",
            },
        ),
        VerifySpec(
            condition=(
                "When purpose is BUILD, content must not resolve the target concept "
                "too early. It should prepare mental material that makes a later "
                "SHIFT needed, not define the whole target concept in advance."
            ),
            applies_when="self.purpose == MovePurpose.BUILD",
            positive_example={
                "context": empty_context,
                "purpose": MovePurpose.BUILD,
                "content": (
                    "The missing rollback is not hard to phrase; it is hard to "
                    "make unavoidable inside the agent loop."
                ),
                "contrast": "sayable vs unavoidable",
                "before": (
                    "The reader sees rollback as something the prompt can mention."
                ),
                "after": (
                    "The reader sees that mentioning rollback and making rollback "
                    "unavoidable are different."
                ),
            },
            negative_example={
                "context": empty_context,
                "purpose": MovePurpose.BUILD,
                "content": (
                    "HDR solves this by representing the whole contract as a Python "
                    "class whose typed fields and semantic verifies must construct "
                    "successfully."
                ),
                "contrast": "informal vs formal",
                "before": "The reader does not know HDR yet.",
                "after": "The reader has already received the full HDR definition.",
            },
        ),
        VerifySpec(
            condition=(
                "When purpose is SHIFT, content changes the reader's interpretation "
                "of already-built material through the contrast axis. It is not "
                "another concrete detail, example, or standalone definition."
            ),
            applies_when="self.purpose == MovePurpose.SHIFT",
            positive_example={
                "context": context_after_build,
                "purpose": MovePurpose.SHIFT,
                "content": (
                    "The problem is not that the agent needs more reminders; the "
                    "problem is that reminders are soft until the harness can "
                    "reject the missing requirement."
                ),
                "contrast": "soft vs enforceable",
                "before": ("The reader reads the rollback miss as a reminder problem."),
                "after": (
                    "The reader reads the rollback miss as an enforceability problem."
                ),
            },
            negative_example={
                "context": context_after_build,
                "purpose": MovePurpose.SHIFT,
                "content": (
                    "The same agent may also forget a verification command, a "
                    "data owner, or a runtime estimate."
                ),
                "contrast": "soft vs enforceable",
                "before": "The reader has seen one missed rollback example.",
                "after": "The reader has seen three more missed fields.",
            },
        ),
        VerifySpec(
            condition=(
                "When purpose is SHIFT, any abstraction in content is earned by "
                "target_reader or prior BUILD moves. The sentence may name a "
                "higher-level lens only when the context has supplied concrete "
                "material that makes the lens immediately legible."
            ),
            applies_when="self.purpose == MovePurpose.SHIFT",
            positive_example={
                "context": context_after_build,
                "purpose": MovePurpose.SHIFT,
                "content": (
                    "That is the contract gap: the prompt contains the requirement, "
                    "but the harness has nothing enforceable to fail."
                ),
                "contrast": "suggestive vs contractual",
                "before": (
                    "The reader sees the rollback miss as a loose instruction problem."
                ),
                "after": (
                    "The reader sees it as a contract gap between what prose asks "
                    "for and what the harness can fail."
                ),
            },
            negative_example={
                "context": empty_context,
                "purpose": MovePurpose.SHIFT,
                "content": (
                    "This is the contract gap between conversational semantics "
                    "and harness-enforced completion."
                ),
                "contrast": "suggestive vs contractual",
                "before": "The reader has not yet seen any concrete failure.",
                "after": "The reader understands a contract gap.",
            },
        ),
        VerifySpec(
            condition=(
                "When purpose is SHIFT, after is an irreversible cognition upgrade "
                "from before. After accepting content, a capable target reader "
                "cannot honestly return to before without ignoring the stated "
                "distinction."
            ),
            applies_when="self.purpose == MovePurpose.SHIFT",
            positive_example={
                "context": context_after_build,
                "purpose": MovePurpose.SHIFT,
                "content": (
                    "A checklist can ask for rollback, but a contract field can make "
                    "rollback absent in the only place that counts: construction."
                ),
                "contrast": "requested vs required",
                "before": (
                    "The reader treats requested rollback and required rollback "
                    "as roughly the same in an agent prompt."
                ),
                "after": (
                    "The reader sees requested rollback and required rollback as "
                    "different completion boundaries."
                ),
            },
            negative_example={
                "context": context_after_build,
                "purpose": MovePurpose.SHIFT,
                "content": (
                    "HDR can make agent workflows more reliable by adding stronger "
                    "validation checks."
                ),
                "contrast": "weak vs strong",
                "before": "The reader thinks validation can help reliability.",
                "after": "The reader thinks stronger validation can help reliability.",
            },
        ),
    ],
)

print("Move contract specification created successfully.")
