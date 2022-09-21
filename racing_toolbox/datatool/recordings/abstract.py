import numpy as np
from abc import ABC, abstractmethod

from racing_toolbox.datatool.models import Recording


class RecorderDataService(ABC):
    @abstractmethod
    def get_recording(
        self, game_name: str, user_name: str, recording_name
    ) -> Recording:
        pass

    @abstractmethod
    def put_observation(
        self,
        image: np.ndarray,
        numerical_data: dict[str, float],
        actions_values: dict[str, float],
    ) -> None:
        pass

    @abstractmethod
    def start_streaming(
        self, game_name: str, user_name: str, recording_name: str, fps: int
    ) -> None:
        pass

    @abstractmethod
    def stop_streaming(self) -> None:
        pass
