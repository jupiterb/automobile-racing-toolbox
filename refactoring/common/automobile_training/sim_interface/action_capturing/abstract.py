from abc import ABC, abstractmethod


class ActionCapturing(ABC):
    @abstractmethod
    def stop(self) -> None:
        """When capturing is stopped, get_actions returns dict with values set to 0.0"""
        pass

    @abstractmethod
    def start(self) -> None:
        """When capturing is started, get_actions returns captured actions"""
        pass

    @abstractmethod
    def get_actions(self) -> dict[str, float]:
        """Returns simulation actions as dict ActiontName -> Value"""
        pass
