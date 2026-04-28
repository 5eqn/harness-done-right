"""
This is the contract of a comprehensive Robot + RL project enhance guidance.
Finish in a separate Python file, it should be run at working directory
`examples/mjlab_enhance_guidance` with command
`python contract/finish_enhance_guidance.py`. Each term
should be implemented in a separate file, expected final output structure:
multiple `*_leaps.py` and a `finish_enhance_guidance.py` inside
`examples/mjlab_enhance_guidance/contract` directory. For each `*_leaps.py`,
invoke a subagent for it. Note that maximum parallel subagent count is 5.
You're allowed to refer to `examples/holomotion_report/contract` and
`examples/mjlab_tracking_report` for deeper understanding of these two repos.
"""
from typing import Self  # type: ignore

from pydantic import model_validator

from hdr.contracts.std import BaseContract, File
from hdr.reviewers import subcato


class Slide(BaseContract):
    content: str


class Highlight(Slide):
    file: File
    text: str  # Prefer multilined evidence unless one line explains EVERYTHING

    @model_validator(mode="after")
    def validate_text_is_unique_file_excerpt(self) -> Self:
        if not self.text:
            raise AssertionError("Highlight text cannot be empty")
        if self.text[0] != "\n":
            raise AssertionError("Highlight text should start with newline")
        if self.text[-1] == "\n":
            raise AssertionError("Highlight should not end with newline")
        padded_content = f"\n{self.file.content}\n"
        padded_text = f"{self.text}\n"
        if padded_text not in padded_content:
            raise AssertionError(
                f"Highlight text must match full line(s) in {self.file.path}"
            )
        if self.file.content.count(self.text) != 1:
            raise AssertionError(
                f"Highlight text must appear exactly once in {self.file.path}"
            )
        return self


class AtomicLeap(BaseContract):
    """
    State an atomic leap from `mjlab` single-motion-tracking to `HoloMotion`
    generalized tracking. `mjlab` scope is motion tracking, `HoloMotion`
    scope is TF-MoE motion tracking.
    """

    # Reference of the related HoloMotion part. This must be a reference to
    # somewhere in `./HoloMotion` subdirectory.
    holomotion_reference: Highlight

    # Reference of the mjlab part waiting to change. This must be a reference
    # to somewhere in `./mjlab` subdirectory.
    mjlab_reference: Highlight

    # How it should be changed. (word description, no code)
    change_direction: str

    # What's the purpose / importance of this change? (word description, no code)
    change_reason: str

    # Draft code of what it should be changed to. (only code)
    changed_code: str

    @model_validator(mode="after")
    def human_review(self) -> Self:
        self.human_expert_review(
            subcato.PROFILE,
            """
            This atomic leap is logically clear, easy to comprehend, serve
            as a necessary part of the whole enhance, and the reference code
            are clearly grounded.""",
        )
        return self


class EnhanceGuidance(BaseContract):
    """
    A comprehensive enhance guidance.
    """

    # How should the deep-learning model be enhanced for generalizable
    # motion-tracking?
    model_leaps: list[AtomicLeap]

    # How should the training process be enhanced?
    training_leaps: list[AtomicLeap]

    # How should the command/data pipeline be enhanced?
    command_leaps: list[AtomicLeap]

    # How should the environment be enhanced?
    environment_leaps: list[AtomicLeap]

    # What terms to be added / removed for observation?
    observation_leaps: list[AtomicLeap]

    # What reward functions should be added / removed?
    reward_leaps: list[AtomicLeap]

    # Name other necessary leaps as needed.
    other_leaps: list[AtomicLeap]

    @model_validator(mode="after")
    def human_review(self) -> Self:
        self.human_expert_review(
            subcato.PROFILE,
            """
            This enhance guidance can be directly given engineers to implement
            the leap, and it covers all major aspects that makes `HoloMotion`
            better from `mjlab` tracking. Differences in just implementation 
            decisions (for example, use issac lab or mjlab) are omitted for
            better clarity for engineers trying to implement generalized motion
            tracking in mjlab environment.""",
        )
        return self