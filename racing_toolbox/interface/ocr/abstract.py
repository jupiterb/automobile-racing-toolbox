import numpy as np
from abc import ABC, abstractmethod


class Ocr(ABC):
    @abstractmethod
    def read_numer(self, image: np.ndarray) -> int:
        pass
    