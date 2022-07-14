import gym
import numpy as np
from typing import Optional, Any

from interface import GameInterface
from interface.models import SteeringAction
from rl.final_state import FinalStateDetector


Frame = Optional[np.ndarray]


class RealTimeEnviroment(gym.Env):
    def __init__(
        self,
        game_interface: GameInterface,
        final_state_detector: FinalStateDetector,
        observation_shape: tuple[int, int, int],
    ) -> None:
        super().__init__()

        self.available_actions: list[list[Optional[SteeringAction]]] = [
            [a] for a in SteeringAction
        ]
        self.available_actions.append([None])  # None means no action
        self.action_space = gym.spaces.Discrete(len(self.available_actions))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=observation_shape,
            dtype=np.uint8,
        )

        self._game_interface = game_interface
        self._final_state_detector = final_state_detector
        self._last_frame: Frame = None

    def reset(self) -> Frame:
        self._game_interface.reset()
        return self._fetch_state()[0]

    def step(self, action: int) -> tuple[Frame, float, bool, dict]:
        self._apply_action(action)

        state, features = self._fetch_state()
        reward = 0  # TODO: reward system

        is_final = self._final_state_detector.is_final(new_features=features)
        self._last_frame = state

        return state, reward, is_final, {}

    def render(self) -> Frame:
        return self._last_frame

    def _apply_action(self, action: int) -> None:
        if self.available_actions[action] != [None]:
            self._game_interface.apply_action(self.available_actions[action])

    def _fetch_state(self) -> tuple[np.ndarray, dict[str, float]]:
        image = self._game_interface.grab_image().astype(np.uint8)
        features = self._game_interface.perform_ocr()
        # TODO: state is image or values vector?
        return image, features
