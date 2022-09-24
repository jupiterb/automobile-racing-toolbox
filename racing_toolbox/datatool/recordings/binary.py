import os
import h5py
import logging
import numpy as np

from racing_toolbox.datatool.recordings import RecorderDataService
from racing_toolbox.datatool.exceptions import ItemExists
from racing_toolbox.datatool.models import Recording


logger = logging.getLogger(__name__)


class BinaryFileRecordingsService(RecorderDataService):

    _tmp_file: h5py.File
    _dts_file: h5py.File

    _tmp_path: str
    _dst_path: str

    _images: h5py.Dataset
    _features: h5py.Dataset
    _actions: h5py.Dataset

    def __init__(self, datasets_dir: str, size_limit: int = 2500) -> None:
        self._datasets_dir = datasets_dir
        self._index = 0
        self._limit = size_limit

    def get_recording(self, game_: str, user: str, recording) -> Recording:
        path = self._get_path_to_file(game_, user, recording)
        if not os.path.exists(path):
            raise ValueError(f"{path} not found!")
        file = h5py.File(path, "r")
        return Recording(
            game=game_,
            user=user,
            name=recording,
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

    def start_streaming(self, game: str, user: str, recording: str, fps: int) -> None:
        self._tmp_path = self._get_path_to_file(game, user, recording, tmp=True)
        self._dst_path = self._get_path_to_file(game, user, recording)

        if os.path.exists(self._dst_path):
            raise ItemExists(game, user, recording)

        dirname = os.path.dirname(self._tmp_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        self._tmp_file = h5py.File(self._tmp_path, "w")
        self._dts_file = h5py.File(self._dst_path, "w")
        self._dts_file.create_dataset("fps", data=np.array([fps]))
        self._index = 0

    def stop_streaming(self) -> None:
        logger.info(
            "Stopping streaming and saving data... It may take long (few minutes)"
        )
        self._move_and_compress_dataset(self._images, "images")
        self._move_and_compress_dataset(self._features, "features")
        self._move_and_compress_dataset(self._actions, "actions")
        self._tmp_file.close()
        self._dts_file.close()
        os.remove(self._tmp_path)
        logger.info("Recording saved successfully.")

    def _get_path_to_file(
        self, game: str, user: str, recording: str, tmp: bool = False
    ) -> str:
        user = f"{user}/tmp" if tmp else user
        return f"{self._datasets_dir}/{game}/{user}/{recording}.hdf5"

    def _move_and_compress_dataset(self, dataset: h5py.Dataset, name: str):
        self._dts_file.create_dataset(
            name,
            data=dataset[: self._index],
            compression="gzip",
            compression_opts=7,
        )
        del dataset

    def _create_datasets(
        self,
        image: np.ndarray,
        numerical_data: dict[str, float],
        actions_values: dict[str, float],
    ):
        self._images = self._tmp_file.create_dataset(
            "images", tuple([self._limit] + list(image.shape)), "i"
        )
        self._features = self._tmp_file.create_dataset(
            "features", (self._limit, len(numerical_data))
        )
        self._actions = self._tmp_file.create_dataset(
            "actions", (self._limit, len(actions_values))
        )
        logging.info("Datasets created")
