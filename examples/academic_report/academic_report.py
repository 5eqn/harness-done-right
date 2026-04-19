from enum import Enum

from hdr.contracts.std import BaseContract
from hdr.contracts.std import Image


class ReferenceType(Enum):
    Article = 1
    Dataset = 2
    Checkpoint = 3
    Repository = 4


class Reference(BaseContract):
    type: ReferenceType
    title: str
    link: str


class Slide(BaseContract):
    """
    A detailed, grounded explanation of something. If it's possible to add
    figures, always add them.
    """

    content: str
    references: list[Reference]
    figures: list[Image]


class ModelArchitecture(Slide):
    """
    Describe a model architecture in detail, from the description, it should be
    possible to rebuild the PyTorch instance of the model.
    """

    pass


class ModelTraining(Slide):
    """
    Describe the training process and requirements of a model, including
    environment, dataset, training GPU-Hour consumption and hyperparameters.
    It should be possible for an experienced ML researcher to run the training
    based solely on the description.
    """

    pass


class Insight(Slide):
    """
    Describe the insight, potentially bridging multiple article references.
    """

    pass


class AcademicReport(BaseContract):
    topic: str
    author: str
    content: list[Slide]
