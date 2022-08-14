from abc import ABC, abstractmethod
from interface.models import SteeringAction


class GameActionCapturing(ABC):
    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def get_captured(self) -> dict[SteeringAction, float]:
        pass
    