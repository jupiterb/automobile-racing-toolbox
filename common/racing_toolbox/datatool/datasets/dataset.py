from abc import ABC, abstractmethod
from contextlib import contextmanager
from pydantic import BaseModel
from numpy import ndarray
from tables import Array
from typing import Union, Generator, Optional


class DatasetModel(BaseModel):
    game: Optional[str] = None
    user: Optional[str] = None
    name: Optional[str] = None
    fps: int
    observations: Union[ndarray, Array]
    actions: Union[ndarray, Array]

    class Config:
        arbitrary_types_allowed = True

    def mergeable_with(self, other):
        return (
            (
                self.game == other.game
                if (self.game is not None and other.game is not None)
                else True
            )
            and self.fps == other.fps
            and self.observations[0].shape == other.observations[0].shape
            and self.actions[0].shape == other.actions[0].shape
        )


class Dataset(ABC):
    @contextmanager
    @abstractmethod
    def get(self) -> Generator[DatasetModel, None, None]:
        pass
