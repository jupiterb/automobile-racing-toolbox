import numpy as np
from abc import ABC, abstractmethod


class AbstractOcr(ABC):
    @abstractmethod
    def read_numer(self, image: np.ndarray) -> int:
        pass
    