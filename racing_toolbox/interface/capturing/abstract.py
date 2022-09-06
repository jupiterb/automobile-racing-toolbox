from abc import ABC, abstractmethod


class GameActionCapturing(ABC):
    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def get_captured(self) -> dict[str, float]:
        pass
    