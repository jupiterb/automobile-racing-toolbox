from abc import ABC, abstractmethod
from interface.models import SteeringAction


class GameActionController(ABC):
    @abstractmethod
    def apply_actions(self, actions: dict[SteeringAction, float]) -> None:
        pass

    @abstractmethod
    def reset_game(self) -> None:
        pass
