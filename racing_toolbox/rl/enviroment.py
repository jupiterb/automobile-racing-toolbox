import gym
import numpy as np
from typing import Optional

from interface import GameInterface
from interface.models import SteeringAction
from rl.final_state import FinalStateDetector

Frame = Optional[np.ndarray]


class RealTimeEnviroment(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self, game_interface: GameInterface, final_state_detector: FinalStateDetector
    ) -> None:
        super().__init__()

        self.available_actions = [
            {SteeringAction.FORWARD},
            {SteeringAction.FORWARD, SteeringAction.LEFT},
            {SteeringAction.FORWARD, SteeringAction.RIGHT},
            {SteeringAction.LEFT},
            {SteeringAction.RIGHT},
            None,
        ]
        self.action_space = gym.spaces.Discrete(len(self.available_actions))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=game_interface.grab_image().shape,
            dtype=np.uint8,
        )

        self._game_interface = game_interface
        self._final_state_detector = final_state_detector
        self._last_frame: Frame = None

    def reset(self) -> Frame:
        self._game_interface.reset(False)
        return self._fetch_state()[0]

    def step(self, action: int) -> tuple[Frame, float, bool, dict]:
        self._apply_action(action)

        state, features = self._fetch_state()
        reward = features["speed"]

        is_final = self._final_state_detector.is_final(new_features=features)
        self._last_frame = state

        if is_final:
            self._final_state_detector.reset()
            print("FINAL!")

        return state, reward, is_final, {}

    def render(self, *args, **kwargs) -> Frame:
        return self._last_frame

    def _apply_action(self, action: int) -> None:
        action_set = self.available_actions[action]
        if action_set != None:
            self._game_interface.apply_action(action_set)

    def _fetch_state(self) -> tuple[np.ndarray, dict[str, float]]:
        image = self._game_interface.grab_image().astype(np.uint8)
        features = self._game_interface.perform_ocr()
        return image, features
