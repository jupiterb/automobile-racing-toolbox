from abc import ABC, abstractmethod
from contextlib import contextmanager
from pydantic import BaseModel
from numpy import ndarray
from tables import Array
from typing import Union, Generator


class DatasetModel(BaseModel):
    game: str
    user: str
    name: str
    fps: int
    observations: Union[ndarray, Array]
    actions: Union[ndarray, Array]

    class Config:
        arbitrary_types_allowed = True

    def mergeable_with(self, other):
        return (
            self.game == other.game
            and self.fps == other.fps
            and self.observations[0].shape == other.observations[0].shape
            and self.actions[0].shape == other.actions[0].shape
        )


class Dataset(ABC):
    @contextmanager
    @abstractmethod
    def get(self) -> Generator[DatasetModel, None, None]:
        pass
