import numpy as np
from interface.ocr.abstract import AbstractOcr


class SevenSegmentsOcr(AbstractOcr):
    def read_numer(self, image: np.ndarray) -> int:
        return 0
