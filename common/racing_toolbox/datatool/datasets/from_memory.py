import os
import tables as tb

from contextlib import contextmanager
from typing import Generator

from racing_toolbox.datatool.datasets import Dataset, DatasetModel
from racing_toolbox.datatool.services import InMemoryDatasetService


class FromMemoryDataset(Dataset):
    def __init__(self, source: str, game: str, user: str, name: str) -> None:
        self._path = InMemoryDatasetService.path_to_file(source, game, user, name)
        self._game = game
        self._user = user
        self._name = name

    @contextmanager
    def get(self) -> Generator[DatasetModel, None, None]:
        if not os.path.exists(self._path):
            raise ValueError(f"{self._path} not found!")
        with tb.File(self._path, driver="H5FD_CORE") as file:
            yield DatasetModel(
                game=self._game,
                user=self._user,
                name=self._name,
                fps=int(file.root.fps[0]),
                observations=file.root.observations,
                actions=file.root.actions,
            )


class LocalDataset(Dataset):
    def __init__(self, path: str):
        self._path = path 

    @contextmanager
    def get(self) -> Generator[DatasetModel, None, None]:
        if not os.path.exists(self._path):
            raise ValueError(f"{self._path} not found!")
        with tb.File(self._path, driver="H5FD_CORE") as file:
            yield DatasetModel(
                fps=int(file.root.fps[0]),
                observations=file.root.observations,
                actions=file.root.actions,
            )