from abc import ABC, abstractmethod


class InputCapturing(ABC):
    @abstractmethod
    def stop(self) -> None:
        """When capturing is stopped, get_inputs returns empty dict (default state)"""
        pass

    @abstractmethod
    def start(self) -> None:
        """When capturing is started, get_inputs returns captured input"""
        pass

    @abstractmethod
    def get_inputs(self) -> dict[str, float]:
        """Returns simulation inputs as dict InputName -> Value"""
        pass
