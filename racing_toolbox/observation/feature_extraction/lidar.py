import numpy as np
from observation.feature_extraction.abstract import FeatureExtractor


class LidarExtractor(FeatureExtractor):
    def extract(self, input: np.ndarray) -> np.ndarray:
        return super().extract(input)
        