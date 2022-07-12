import gym
import numpy as np
from typing import Optional

from interface import GameInterface
from observation import ObservationCreator
from rl.final_state import FinalStateDetector


Frame = Optional[np.ndarray]


class RealTimeEnviroment(gym.Env):
    def __init__(
        self,
        game_interface: GameInterface,
        observation_creator: ObservationCreator,
        final_state_detector: FinalStateDetector,
    ) -> None:
        super().__init__()
        self._game_interface = game_interface
        self._observation_creator = observation_creator
        self._final_state_detector = final_state_detector
        self._last_frame: Frame = None

    def reset(self) -> Frame:
        self._game_interface.reset()
        return self._fetch_state()[0]

    def step(self, action: int) -> tuple[Frame, float, bool, dict]:
        self._apply_action(action)

        state, _, features = self._fetch_state()
        reward = 0  # TODO: reward system

        is_final = self._final_state_detector.is_final(new_features=features)
        self._last_frame = state

        return state, reward, is_final, {}

    def render(self) -> Frame:
        return self._last_frame

    def _apply_action(self, action: int) -> None:
        pass

    def _fetch_state(self) -> tuple[Frame, np.ndarray, dict[str, float]]:
        image = self._game_interface.grab_image()
        features = self._game_interface.perform_ocr()
        self._observation_creator.add_to_buffer(image, np.array(features.values()))
        observation = self._observation_creator.get_observation()
        # TODO: state is image or values vector?
        return observation[0] if observation else None, image, features
