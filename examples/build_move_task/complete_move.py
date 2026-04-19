from __future__ import annotations

from hdr.tasks.coding import MarkdownFile

from move import Move
from move_support import MoveContext, MovePurpose, MoveSnapshot


target_reader = MarkdownFile(path="hdr_target_reader.md")
target_meaning = MarkdownFile(path="hdr_target_meaning.md")

prior_moves: list[MoveSnapshot] = []


def move_context() -> MoveContext:
    return MoveContext(
        target_reader=target_reader,
        target_concept="HDR",
        target_meaning=target_meaning,
        prior_moves=[*prior_moves],
    )


def choose_move(
    *,
    purpose: MovePurpose,
    content: str,
    contrast: str,
    before: str,
    after: str,
) -> Move:
    move_contract = Move(
        context=move_context(),
        purpose=purpose,
        content=content,
        contrast=contrast,
        before=before,
        after=after,
    )
    prior_moves.append(
        MoveSnapshot(
            purpose=move_contract.purpose,
            content=move_contract.content,
            contrast=move_contract.contrast,
            before=move_contract.before,
            after=move_contract.after,
        )
    )
    return move_contract


move_steps = [
    choose_move(
        purpose=MovePurpose.BUILD,
        content=(
            "A migration prompt can name rollback, verification commands, and "
            "owner signoff, yet those nouns remain lines of text the harness "
            "never has to inspect before accepting done."
        ),
        contrast="named vs inspected",
        before=(
            "The reader treats named requirements in a detailed prompt as "
            "meaningful coverage for agent work."
        ),
        after=(
            "The reader sees that naming requirements in prose is weaker than "
            "making them inspectable by the harness."
        ),
    ),
    choose_move(
        purpose=MovePurpose.SHIFT,
        content=(
            "That is the contract gap: the prompt contains the requirement, "
            "but the harness has nothing enforceable to fail."
        ),
        contrast="suggestive vs contractual",
        before=(
            "The reader sees the missed requirement as a loose instruction problem."
        ),
        after=(
            "The reader sees it as a contract gap between what prose suggests "
            "and what the harness can reject."
        ),
    ),
    choose_move(
        purpose=MovePurpose.SHIFT,
        content=(
            "HDR closes that gap by making the intended outcome a Python contract "
            "object, so completion means constructing the object rather than "
            "merely declaring the prose satisfied."
        ),
        contrast="declared vs constructed",
        before=(
            "The reader treats completion as the agent's claim that it followed "
            "the prompt."
        ),
        after=(
            "The reader treats completion as successful construction of an "
            "explicit contract object."
        ),
    ),
    choose_move(
        purpose=MovePurpose.BUILD,
        content=(
            "Typed fields force required evidence into named slots, so a "
            "missing rollback plan is absent from data the harness can inspect, "
            "not merely absent from a paragraph."
        ),
        contrast="implicit vs typed",
        before=(
            "The reader thinks a requirement can live implicitly inside a prose "
            "paragraph."
        ),
        after=(
            "The reader sees that a requirement becomes inspectable when it is "
            "held in a typed field."
        ),
    ),
    choose_move(
        purpose=MovePurpose.BUILD,
        content=(
            "Programmatic checks reject malformed shape at construction time, "
            "before any semantic judgment about whether the agent's answer is "
            "good enough."
        ),
        contrast="malformed vs well-shaped",
        before=(
            "The reader treats validation as a later quality judgment over the "
            "agent's answer."
        ),
        after=(
            "The reader sees that construction can reject the wrong shape "
            "before semantic review begins."
        ),
    ),
    choose_move(
        purpose=MovePurpose.SHIFT,
        content=(
            "Then semantic self.verify() checks reject missing meaning, turning "
            "the remaining question from 'did the agent sound careful?' into "
            "'does this constructed contract actually satisfy the intended meaning?'"
        ),
        contrast="performative vs semantic",
        before=(
            "The reader judges the agent by whether the response sounds careful "
            "after the shape checks pass."
        ),
        after=(
            "The reader judges the constructed contract by whether it satisfies the "
            "intended meaning."
        ),
    ),
    choose_move(
        purpose=MovePurpose.SHIFT,
        content=(
            "Because the run is not complete until the final contract instance "
            "constructs successfully, the agent has to keep working where a "
            "prose-only instruction could have stopped at a plausible answer."
        ),
        contrast="plausible vs complete",
        before=(
            "The reader treats a plausible answer to the prompt as the natural "
            "stopping point."
        ),
        after=(
            "The reader sees successful contract construction as the stopping point "
            "that makes the agent keep working."
        ),
    ),
]

move_contract = move_steps[-1]

print(f"Move chain completed with {len(move_steps)} atomic moves.")
