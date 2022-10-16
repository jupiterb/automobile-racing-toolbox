import numpy as np
from abc import ABC, abstractmethod
from racing_toolbox.observation.utils import ScreenFrame
from racing_toolbox.observation.utils.ocr.config import OcrConfiguration


class OcrTool:
    def __init__(
        self, config: dict[str, tuple[ScreenFrame, OcrConfiguration]], ocr_cls
    ) -> None:
        self._ocrs = [
            (name, frame, ocr_cls(ocr_config))
            for name, (frame, ocr_config) in config.items()
        ]

    def perform(self, image: np.ndarray) -> dict[str, float]:
        return {
            name: float(ocr.read_number(frame.apply(image)))
            for name, frame, ocr in self._ocrs
        }


class Ocr(ABC):
    @abstractmethod
    def read_number(self, image: np.ndarray) -> int:
        pass
