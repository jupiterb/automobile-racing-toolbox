import gym
import gym.spaces 
import numpy as np
from typing import Optional

from racing_toolbox.interface import GameInterface
from racing_toolbox.environment.final_state import FinalStateDetector
from racing_toolbox.observation.utils.ocr import OcrTool
from logging import getLogger

logger = getLogger(__name__)
Frame = Optional[np.ndarray]


class RealTimeEnviroment(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        game_interface: GameInterface,
        ocr_tool: OcrTool,
        final_state_detector: FinalStateDetector,
    ) -> None:
        super().__init__()

        self._available_actions = game_interface.get_possible_actions()

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=[len(self._available_actions)],
            dtype=np.float16,
        )

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=game_interface.screen.shape,
            dtype=np.uint8,
        )

        self._game_interface = game_interface
        self._ocr = ocr_tool
        self._final_state_detector = final_state_detector
        self._last_frame: Frame = None

    def reset(self) -> Frame:
        self._game_interface.reset()
        return self._fetch_state()[0]

    def step(self, actions: np.ndarray) -> tuple[Frame, float, bool, dict]:
        self._apply_action(actions)

        state, features = self._fetch_state()
        reward = features["speed"]
        logger.debug(f"current speed: {reward}")

        is_final = self._final_state_detector.is_final(new_features=features)
        self._last_frame = state

        if is_final:
            self._final_state_detector.reset()
            print("FINAL!")

        return state, reward, is_final, {"speed": reward}

    def render(self, *args, **kwargs) -> Frame:
        return self._last_frame

    def _apply_action(self, actions: np.ndarray) -> None:
        actions_values = {
            self._available_actions[action]: value
            for action, value in enumerate(actions)
        }
        self._game_interface.apply_action(actions_values)

    def _fetch_state(self) -> tuple[np.ndarray, dict[str, float]]:
        image = self._game_interface.grab_image().astype(np.uint8)
        features = self._ocr.perform(image)
        return image, features
