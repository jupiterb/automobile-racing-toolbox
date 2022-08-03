import gym
import numpy as np
from typing import Optional

from interface import GameInterface
from interface.config import SteeringAction
from rl.event import EventDetector

Frame = Optional[np.ndarray]


class RealTimeEnvironment(gym.Env):
    def __init__(
        self, game_interface: GameInterface, final_state_detector: EventDetector
    ) -> None:
        super().__init__()

        self.available_actions: list[list[Optional[SteeringAction]]] = [
            [SteeringAction.FORWARD],
            [SteeringAction.FORWARD, SteeringAction.LEFT],
            [SteeringAction.FORWARD, SteeringAction.RIGHT],
            [SteeringAction.BREAK, SteeringAction.RIGHT],
            [SteeringAction.BREAK, SteeringAction.RIGHT],
            [SteeringAction.LEFT],
            [SteeringAction.RIGHT],
            [None],
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
        self._game_interface.reset()
        return self._fetch_state()[0]

    def step(self, action: int) -> tuple[Frame, float, bool, dict]:
        self._apply_action(action)

        state, features = self._fetch_state()
        reward = features["speed"]

        is_final = self._final_state_detector.is_final(features)
        self._last_frame = state

        if is_final:
            self._final_state_detector.reset()
            print("FINAL!")

        return state, reward, is_final, {}

    def render(self) -> Frame:
        return self._last_frame

    def _apply_action(self, action: int) -> None:
        if self.available_actions[action] != [None]:
            self._game_interface.apply_action(self.available_actions[action])

    def _fetch_state(self) -> tuple[np.ndarray, dict[str, float]]:
        image = self._game_interface.grab_image().astype(np.uint8)
        features = self._game_interface.perform_ocr()
        return image, features
