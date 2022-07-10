import numpy as np
from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, input: np.ndarray) -> np.ndarray:
        pass
    