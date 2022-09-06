from abc import ABC, abstractmethod


class GameActionController(ABC):
    @abstractmethod
    def apply_actions(self, actions: dict[str, float]) -> None:
        pass

    @abstractmethod
    def reset_game(self) -> None:
        pass

    @abstractmethod
    def get_possible_actions(self) -> list[str]:
        pass
