import numpy as np
from typing import Generator

from racing_toolbox.datatool.models import Recording


class RecordingsContainer:
    def __init__(self) -> None:
        self._recordings: list[Recording] = []

    def try_add(self, recording: Recording) -> bool:
        if self.can_be_added(recording):
            self._recordings.append(recording)
            return True
        return False

    def can_be_added(self, recording: Recording) -> bool:
        return False

    def get_all(
        self,
    ) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Yields tuples of images, features and actions
        """
        for recording in self._recordings:
            for image, features, actions in zip(
                recording.images, recording.features, recording.actions
            ):
                yield image, features, actions
