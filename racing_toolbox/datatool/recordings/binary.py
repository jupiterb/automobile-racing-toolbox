import os
import h5py
import numpy as np
from torch import chunk

from racing_toolbox.datatool.recordings import RecorderDataService
from racing_toolbox.datatool.exceptions import ItemExists
from racing_toolbox.datatool.models import Recording


class BinaryFileRecordingsService(RecorderDataService):

    _max_size = 50_000

    _file: h5py.File

    _images: h5py.Dataset
    _features: h5py.Dataset
    _actions: h5py.Dataset

    def __init__(self, datasets_dir: str) -> None:
        self._datasets_dir = datasets_dir
        self._index = 0

    def get_recording(
        self, game_name: str, user_name: str, recording_name
    ) -> Recording:
        path = self._get_path_to_file(game_name, user_name, recording_name)
        if not os.path.exists(path):
            raise ValueError(f"{path} not found!")
        file = h5py.File(path, "r")
        return Recording(
            game=game_name,
            user=user_name,
            name=recording_name,
            fps=int(file["fps"][0]),
            images=file["images"],
            actions=file["actions"],
            features=file["features"],
        )

    def put_observation(
        self,
        image: np.ndarray,
        numerical_data: dict[str, float],
        actions_values: dict[str, float],
    ) -> None:
        if not self._index:
            self._create_datasets(image, numerical_data, actions_values)
        self._images[self._index, ...] = image
        self._features[self._index, ...] = np.array(list(numerical_data.values()))
        self._actions[self._index, ...] = np.array(list(actions_values.values()))
        self._index += 1

    def start_streaming(
        self, game_name: str, user_name: str, recording_name: str, fps: int
    ) -> None:
        path = self._get_path_to_file(game_name, user_name, recording_name)
        if os.path.exists(path):
            raise ItemExists(game_name, user_name, recording_name)
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self._file = h5py.File(path, "w")
        self._file.create_dataset("fps", data=np.array([fps]))
        self._index = 0

    def stop_streaming(self) -> None:
        size = self._index
        self._images.resize(size, 0)
        self._features.resize(size, 0)
        self._actions.resize(size, 0)
        self._file.close()

    def _get_path_to_file(
        self, game_name: str, user_name: str, recording_name: str
    ) -> str:
        return f"{self._datasets_dir}/{game_name}/{user_name}/{recording_name}.hdf5"

    def _create_datasets(
        self,
        image: np.ndarray,
        numerical_data: dict[str, float],
        actions_values: dict[str, float],
    ):
        limit = BinaryFileRecordingsService._max_size
        self._images = self._file.create_dataset(
            "images", tuple([limit] + list(image.shape)), "i", chunks=True
        )
        self._features = self._file.create_dataset(
            "features", (limit, len(numerical_data)), chunks=True
        )
        self._actions = self._file.create_dataset(
            "actions", (limit, len(actions_values)), chunks=True
        )
