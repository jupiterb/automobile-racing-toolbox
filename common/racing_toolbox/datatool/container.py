import numpy as np
from typing import Generator, Optional

from racing_toolbox.datatool.datasets import Dataset


class DatasetContainer:
    def __init__(self) -> None:
        self._datasets: list[Dataset] = []

    def try_add(self, dataset: Dataset) -> bool:
        if self.can_be_added(dataset):
            self._datasets.append(dataset)
            return True
        return False

    def can_be_added(self, dataset: Dataset) -> bool:
        if not any(self._datasets):
            return True
        result = False
        with self._datasets[0].get() as representant:
            with dataset.get() as other:
                result = representant.mergeable_with(other)
        return result

    @property
    def fps(self) -> Optional[int]:
        if not any(self._datasets):
            return None
        with self._datasets[0].get() as representant:
            fps = representant.fps
        return fps

    def get_all(
        self,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        for dataset in self._datasets:
            with dataset.get() as model:
                for observation, actions in zip(model.observations, model.actions):
                    yield observation, actions

