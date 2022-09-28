import os
import tables as tb
import numpy as np
import multiprocessing as mp

from contextlib import AbstractContextManager
from types import TracebackType
from typing import Type

from racing_toolbox.datatool.exceptions import ItemExists
from racing_toolbox.datatool.services import AbstractDatasetService


class InMemoryDatasetConsumer(AbstractContextManager):

    _file: tb.File
    _observations: tb.EArray
    _actions: tb.EArray

    def __init__(self, path_to_file: str, fps: int, queue: mp.Queue) -> None:
        self._path = path_to_file
        self._fps = fps
        self._queue = queue

    def __enter__(self):
        self._file = tb.File(self._path, "w", driver="H5FD_CORE")
        return self

    def __exit__(
        self,
        __exc_type: Type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool | None:
        self._file.close()

    def consume(self):
        start = True
        item = self._queue.get()
        while item is not False:
            print("New item!")
            observations, actions = item
            if start:
                self._create_arrays(observations, actions)
                start = False
            self._observations.append(observations)
            self._actions.append(actions)
            item = self._queue.get()

    def _create_arrays(self, observation: np.ndarray, actions: np.ndarray):
        self._observations = self._file.create_earray(
            self._file.root,
            "observations",
            tb.Int8Atom(),
            tuple([0] + list(observation.shape)[1:]),
            "Obs",
            chunkshape=observation.shape,
        )
        self._actions = self._file.create_earray(
            self._file.root,
            "actions",
            tb.Float16Atom(),
            tuple([0] + list(actions.shape)[1:]),
            "Act",
            chunkshape=actions.shape,
        )
        self._file.create_array(self._file.root, "fps", np.array([self._fps]))


class InMemoryDatasetService(AbstractDatasetService):

    _consumer: mp.Process
    _observations: np.ndarray
    _actions: np.ndarray

    def __init__(
        self,
        destination: str,
        game: str,
        user: str,
        dataset: str,
        fps: int,
        batch_size: int = 25,
    ) -> None:
        self._path = InMemoryDatasetService.path_to_file(
            destination, game, user, dataset
        )
        if os.path.exists(self._path):
            raise ItemExists(game, user, dataset)
        self._index = 0
        self._queue = mp.Queue()
        self._fps = fps
        self._batch_size = batch_size

    def __enter__(self) -> AbstractDatasetService:
        dirname = os.path.dirname(self._path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self._consumer = mp.Process(target=self._consuming, args=())
        self._consumer.start()
        return self

    def __exit__(
        self,
        __exc_type: Type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool | None:
        self._queue.put(False)
        self._consumer.join()

    def put(self, observation: np.ndarray, actions: dict[str, float]) -> None:
        actions_values = np.array(list(actions.values()))
        if not self._index:
            self._create_batches(observation, actions_values)
        self._observations[self._index] = observation
        self._actions[self._index] = actions_values
        self._index += 1
        if self._index == self._batch_size:
            self._queue.put((self._observations, self._actions))
            self._index = 0

    @staticmethod
    def path_to_file(
        root: str, game_name: str, user_name: str, dataset_name: str
    ) -> str:
        return f"{root}/{game_name}/{user_name}/{dataset_name}.h5"

    def _consuming(self):
        with InMemoryDatasetConsumer(self._path, self._fps, self._queue) as consumer:
            consumer.consume()

    def _create_batches(self, observation: np.ndarray, actions: np.ndarray) -> None:
        self._observations = np.zeros(
            tuple([self._batch_size] + list(observation.shape))
        )
        self._actions = np.zeros(tuple([self._batch_size] + list(actions.shape)))
