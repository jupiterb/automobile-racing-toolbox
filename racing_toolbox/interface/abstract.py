from abc import ABC, abstractmethod, abstractproperty
import numpy as np

from interface.models import SteeringAction


class GameInterface(ABC):
    @abstractproperty
    def screen_shape(self) -> tuple[int, int, int]:
        pass

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
    def apply_action(self, discrete_actions: list[SteeringAction]) -> None:
        pass

    @abstractmethod
    def read_action(self) -> list[SteeringAction]:
        pass
