from abc import ABC, abstractmethod
import numpy as np

from interface.models import SteeringAction


class GameInterface(ABC):
    @abstractmethod
    def restart(self) -> None:
        pass

    @abstractmethod
    def grab_image(self) -> np.ndarray:
        pass

    @abstractmethod
    def perform_ocr(self, on_last_image: bool = True) -> list[int]:
        pass

    @abstractmethod
    def apply_action(self, discrete_actions: list[SteeringAction]) -> None:
        pass

    @abstractmethod
    def read_action(self) -> list[SteeringAction]:
        pass
