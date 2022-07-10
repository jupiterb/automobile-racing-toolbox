import numpy as np
from observation.feature_extraction.abstract import FeatureExtractor


class AutoencoderExtractor(FeatureExtractor):
    def extract(self, input: np.ndarray) -> np.ndarray:
        return super().extract(input)
        