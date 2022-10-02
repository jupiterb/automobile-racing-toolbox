import os
import tables as tb
import numpy as np
import multiprocessing as mp

from contextlib import AbstractContextManager
from types import TracebackType
from typing import Type, Optional

from racing_toolbox.datatool.exceptions import ItemExists
from racing_toolbox.datatool.services import AbstractDatasetService


class InMemoryDatasetConsumer(AbstractContextManager):

    _file: tb.File
    _observations_array: tb.EArray
    _actions_array: tb.EArray
    _observations_batch: np.ndarray
    _actions_batch: np.ndarray

    def __init__(
        self, path_to_file: str, fps: int, queue: mp.Queue, batch_size: int
    ) -> None:
        self._path = path_to_file
        self._fps = fps
        self._queue = queue
        self._batch_size = batch_size
        self._index = 0

    def __enter__(self):
        self._file = tb.File(self._path, "w", driver="H5FD_CORE")
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> bool | None:
        self._file.close()

    def consume(self):
        item = self._queue.get()
        while item is not False:
            observation, action = item
            if not self._index:
                self._init_arrays(observation.shape, action.shape)
            self._update_batches(observation, action)
            if not self._index % self._batch_size:
                self._observations_array.append(self._observations_batch)
                self._actions_array.append(self._actions_batch)
            item = self._queue.get()

    def _update_batches(self, observation: np.ndarray, action: np.ndarray):
        batch_index = self._index % self._batch_size
        self._observations_batch[batch_index] = observation
        self._actions_batch[batch_index] = action
        self._index += 1

    def _init_arrays(self, observation_shape: tuple, action_shape: tuple):
        def create(shape: tuple, name: str, atom: tb.Atom):
            batch = np.zeros(tuple([self._batch_size] + list(shape)))
            array_shape = tuple([0] + list(shape))
            array = self._file.create_earray(
                self._file.root, name, atom, array_shape, name, chunkshape=batch.shape
            )
            return batch, array

        self._observations_batch, self._observations_array = create(
            observation_shape, "observations", tb.Int8Atom()
        )
        self._actions_batch, self._actions_array = create(
            action_shape, "actions", tb.Float16Atom()
        )
        self._file.create_array(self._file.root, "fps", np.array([self._fps]))


class InMemoryDatasetService(AbstractDatasetService):

    _consumer: mp.Process

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
        self._queue.put((observation, actions_values))

    @staticmethod
    def path_to_file(
        root: str, game_name: str, user_name: str, dataset_name: str
    ) -> str:
        return f"{root}/{game_name}/{user_name}/{dataset_name}.h5"

    def _consuming(self):
        with InMemoryDatasetConsumer(
            self._path, self._fps, self._queue, self._batch_size
        ) as consumer:
            consumer.consume()
