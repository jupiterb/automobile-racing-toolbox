from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from enum import Enum

_Action = TypeVar("_Action", bound=Enum, covariant=True)


class GameActionController(ABC, Generic[_Action]):
    def __init__(
        self, action_mapping: dict[str, _Action], reset_sequence: list[_Action]
    ) -> None:
        self._action_mapping = action_mapping
        self._reset_sequence = reset_sequence

    def get_possible_actions(self) -> list[str]:
        return list(self._action_mapping)

    @abstractmethod
    def apply_actions(self, actions: dict[str, float]) -> None:
        pass

    @abstractmethod
    def reset_game(self) -> None:
        pass
