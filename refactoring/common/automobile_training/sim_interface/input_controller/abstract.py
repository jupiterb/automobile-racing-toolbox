from abc import ABC, abstractmethod


class InputController(ABC):
    @property
    @abstractmethod
    def possible_inputs(self) -> set[str]:
        """Inputs with keys in this set can only be applied"""
        pass

    @abstractmethod
    def apply(self, inputs: dict[str, float]) -> None:
        """Apply inputs, which are dict InputName -> Value"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets simulation to default state"""
        pass
