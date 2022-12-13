import numpy as np
from pydantic import BaseModel


class ScreenFrame(BaseModel):
    top: float = 0.0
    bottom: float = 1.0
    left: float = 0.0
    right: float = 1.0

    def apply(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        top, bottom = int(self.top * height), int(self.bottom * height)
        left, right = int(self.left * width), int(self.right * width)
        return image[top:bottom, left:right]
