import numpy as np
from abc import ABC, abstractmethod


class AbstractOcr(ABC):
    @abstractmethod
    def read_number(self, image: np.ndarray) -> int:
        pass
    