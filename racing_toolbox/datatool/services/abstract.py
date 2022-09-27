from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from numpy import ndarray


class AbstractDatasetService(AbstractContextManager, ABC):
    def __init__(
        self,
        destination: str,
        game_name: str,
        user_name: str,
        dataset_name: str,
        fps: int,
    ) -> None:
        super().__init__()

    @abstractmethod
    def put(self, observation: ndarray, actions: dict[str, float]) -> None:
        pass
