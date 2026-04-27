"""
This is the contract of a comprehensive Robot + RL project analysis.
Finish in `finish_project_analysis.py`, it should be run at working directory
`../` with command `python contract/finish_project_analysis.py`.
"""
import os
from typing import Self  # type: ignore

from pydantic import model_validator, Field

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


class ProofFromCode(BaseContract):
    """
    Proves that a workspace has some functionality from grounded code evidence.

    Prefer bottom-up explanation: say why the highlighted lines prove the claim,
    grounded down to the concrete source/API/config level that actually makes
    the claim true. Highlights can not be an empty list.
    """

    highlights: list["Highlight | ProofFromCode"]

    @model_validator(mode="after")
    def validate_has_highlights(self) -> Self:
        if not self.highlights:
            raise AssertionError("ProofFromCode must include at least one highlight")
        return self

    @model_validator(mode="after")
    def human_review(self) -> Self:
        self.human_expert_review(
            subcato.PROFILE,
            """
            This proof is grounded to the deepest implementation details
            and does not miss any single part in logic chain.""",
        )
        return self


class ProjectAnalysis(BaseContract):
    """
    A comprehensive analysis of a robot RL work. Each term must be implemented in a seperate file, then use a `finish_project_analysis.py` to combine them all.
    """

    # Proof that scene / environment is constructed correctly, the environment should make sense for a stable training, you should inspect what the environment actually consists of.
    scene_correct: ProofFromCode

    # Proof that the code uses the correct model that will be trained with RL, with detailed information about the model type (MLP / Transformer / Other), layer architecture (hidden dimensions, etc.) and other things (activation function, etc.) that affects the code of the model.
    model_correct: ProofFromCode

    # Proof that actions are of the correct shape and applied to the environment correctly, with detailed info about each term of action.
    action_correct: ProofFromCode

    # Proof that observations are collected correctly from the environment, with detailed info about each term of observation, the dimension / length of each type of observation and it's meaning.
    observation_correct: ProofFromCode

    # Proof that environment reset sets the scene in a correct / stable way that supports robust training later on.
    reset_correct: ProofFromCode

    # Proof that reward is calculated correctly, with detailed info about each term of reward, how they're collected from the environment, and what they mean / what they're trying to do.
    reward_correct: ProofFromCode

    # Proof that loss term is calculated correctly.
    loss_correct: ProofFromCode

    # Proof that trainer is launched and updates the model / environment correctly.
    trainer_correct: ProofFromCode

    # Proof that training logs (rewards / steps / etc.) are done correctly.
    logs_correct: ProofFromCode

    # Proof that checkpoint is saved and loaded correctly.
    checkpoint_correct: ProofFromCode

    # Proof that viewer can be launched with proper script / method.
    viewer_correct: ProofFromCode

    # Proof that (motion) command is built and applied correctly. It should be very detailed, and focus on how it loops/randoms from the dataset, and how randomly-generated/curriculum-generated (if exists) command actually applies to the RL training / evaluation process.
    command_correct: ProofFromCode
