from abc import ABC, abstractmethod


class ActionController(ABC):
    @property
    @abstractmethod
    def possible_actions(self) -> set[str]:
        """Actions with keys in this set can only be applied"""
        pass

    @abstractmethod
    def apply(self, actions: dict[str, float]) -> None:
        """Applies actions, which are dict ActionName -> Value"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets simulation to default state"""
        pass
