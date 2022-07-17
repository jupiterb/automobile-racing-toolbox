import gym
import numpy as np
from typing import Optional, Any

from interface import GameInterface
from interface.models import SteeringAction
from rl.final_state import FinalStateDetector
import cv2

Frame = Optional[np.ndarray]


def _to_grayscale(frame):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    grayscale_image = np.dot(frame, rgb_weights)
    return grayscale_image


def _rescale(frame, max_side_len):
    scale = max_side_len / np.max(frame.shape)
    target_shape = list((scale * np.array(frame.shape)).astype(np.uint8))
    return cv2.resize(frame, dsize=target_shape[::-1], interpolation=cv2.INTER_CUBIC)


class RealTimeEnviroment(gym.Env):
    def __init__(
        self,
        game_interface: GameInterface,
        final_state_detector: FinalStateDetector,
        observation_shape: tuple[int, int],
    ) -> None:
        super().__init__()

        self.available_actions: list[list[Optional[SteeringAction]]] = [
            [SteeringAction.FORWARD],
            [SteeringAction.LEFT],
            [SteeringAction.RIGHT],
            [None],
        ]
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
        reward = features["speed"]  # TODO: reward system

        is_final = self._final_state_detector.is_final(new_features=features)
        self._last_frame = state

        if is_final:
            self._final_state_detector.reset()
            print("FINAL!")
        else:
            print("Reward:", reward)
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
        image = _rescale(_to_grayscale(image), 100).astype(np.uint8)
        return image, features
