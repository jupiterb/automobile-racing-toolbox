import numpy as np
from abc import ABC, abstractmethod


class AbstractOcr(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def read_numer(self, image: np.ndarray) -> int:
        pass
    