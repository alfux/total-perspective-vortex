"""Trainer class is responsible for the MLP classification training."""

from typing import Self

from numpy import ndarray


class Trainer:
    """Trainer class' instances hold configuration and training sessions."""

    def __init__(self: Self, input: ndarray, truth: ndarray) -> None:
        """Configure the training session."""
        pass
