import os
import tables as tb
import numpy as np

from contextlib import contextmanager
from types import TracebackType
from typing import Type, Generator

from racing_toolbox.datatool.exceptions import ItemExists
from racing_toolbox.datatool.datasets import Dataset, DatasetModel
from racing_toolbox.datatool.services import AbstractDatasetService


class InMemoryDatasetService(AbstractDatasetService):

    _file: tb.File
    _observations: tb.EArray
    _actions: tb.EArray

    class FromMemoryDataset(Dataset):
        def __init__(self, path: str, game: str, user: str, name: str) -> None:
            self._path = path
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

    def __init__(
        self, destination: str, game: str, user: str, dataset: str, fps: int
    ) -> None:
        self._path = InMemoryDatasetService._path_to_file(
            destination, game, user, dataset
        )
        if os.path.exists(self._path):
            raise ItemExists(game, user, dataset)
        self._start = True
        self._fps = fps

    def __enter__(self) -> AbstractDatasetService:
        dirname = os.path.dirname(self._path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self._file = tb.open_file(self._path, "w", driver="H5FD_CORE")
        return self

    def __exit__(
        self,
        __exc_type: Type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool | None:
        self._file.close()

    def put(self, observation: np.ndarray, actions: dict[str, float]) -> None:
        actions_values = np.array(list(actions.values()))
        if self._start:
            self._create_arrays(observation, actions_values)
            self._start = False
        self._observations.append(np.array([observation]))
        self._actions.append(np.array([actions_values]))

    @staticmethod
    def get_dataset(source: str, game: str, user: str, dataset: str) -> Dataset:
        path = InMemoryDatasetService._path_to_file(source, game, user, dataset)
        return InMemoryDatasetService.FromMemoryDataset(path, game, user, dataset)

    @staticmethod
    def _path_to_file(
        root: str, game_name: str, user_name: str, dataset_name: str
    ) -> str:
        return f"{root}/{game_name}/{user_name}/{dataset_name}.h5"

    def _create_arrays(self, observation: np.ndarray, actions: np.ndarray) -> None:
        self._observations = self._file.create_earray(
            self._file.root,
            "observations",
            tb.Int8Atom(),
            tuple([0] + list(observation.shape)),
            "Obs",
        )
        self._actions = self._file.create_earray(
            self._file.root,
            "actions",
            tb.Float16Atom(),
            tuple([0] + list(actions.shape)),
            "Act",
        )
        self._file.create_array(self._file.root, "fps", np.array([self._fps]))
