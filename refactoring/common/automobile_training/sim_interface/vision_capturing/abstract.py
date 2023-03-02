from abc import ABC, abstractmethod
import numpy as np


class SimVisionNotFoundException(Exception):
    def __init__(self, description: str) -> None:
        super().__init__(description)


class VisionCapturing(ABC):
    def __init__(self, height: int, width: int) -> None:
        """
        Arguments:
        height - 1st dimension of vision shape
        width - 2nd dimension of vision shape
        """
        self._size = height, width

    @property
    def shape(self) -> tuple[int, int, int]:
        """get_vision method returns numpy array with such a shape"""
        return *self._size, 3

    @abstractmethod
    def get_vision(self) -> np.ndarray:
        """Returns numpy array with simulation vision"""
        pass
