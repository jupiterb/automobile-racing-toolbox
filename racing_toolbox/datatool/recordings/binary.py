import numpy as np

from racing_toolbox.datatool.recordings import RecorderDataService
from racing_toolbox.datatool.exceptions import ItemExists
from racing_toolbox.datatool.models import Recording


class BinaryFileRecordingsService(RecorderDataService):
    def __init__(self) -> None:
        super().__init__()

    def get_recording(
        self, game_name: str, user_name: str, recording_name
    ) -> Recording:
        return super().get_recording(game_name, user_name, recording_name)

    def put_observation(
        self,
        image: np.ndarray,
        numerical_data: dict[str, float],
        actions_values: dict[str, float],
    ) -> None:
        return super().put_observation(image, numerical_data, actions_values)

    def start_streaming(
        self, game_name: str, user_name: str, recording_name: str, fps: int
    ) -> None:
        return super().start_streaming(game_name, user_name, recording_name, fps)

    def stop_streaming(self) -> None:
        return super().stop_streaming()
