from abc import ABC, abstractmethod
import numpy as np

from interface.models import SteeringAction


class GameInterface(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def grab_image(self) -> np.ndarray:
        pass

    @abstractmethod
    def perform_ocr(self, on_last_image: bool = True) -> dict[str, float]:
        pass

    @abstractmethod
    def apply_action(self, discrete_actions: dict[SteeringAction, float]) -> None:
        pass

    @abstractmethod
    def read_action(self) -> dict[SteeringAction, float]:
        pass
