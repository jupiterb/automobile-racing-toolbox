from typing import Optional
import numpy as np
import time

from racing_toolbox.interface.screen import ScreenProvider
from racing_toolbox.interface.capturing import GameActionCapturing
from racing_toolbox.interface.controllers import GameActionController
from racing_toolbox.observation.utils.screen_frame import ScreenFrame


class GameInterface:
    def __init__(
        self,
        game_id: str,
        reset_seconds: int,
        screen: ScreenProvider,
        controller: Optional[GameActionController] = None,
        capturing: Optional[GameActionCapturing] = None,
    ) -> None:
        self._name = game_id
        self._screen = screen
        self._reset_seconds = reset_seconds
        self._controller = controller
        self._capturing = capturing

    @property
    def screen(self) -> ScreenProvider:
        return self._screen

    def name(self) -> str:
        return self._name

    def reset(self, enable_action_read: bool = True) -> None:
        if self._controller:
            self._controller.reset_game()
        time.sleep(self._reset_seconds)
        if self._capturing:
            self._capturing.stop()
            if enable_action_read:
                self._capturing.start()

    def grab_image(self) -> np.ndarray:
        return self._screen.grab_image()

    def apply_action(self, actions: dict[str, float]) -> None:
        if self._controller:
            self._controller.apply_actions(actions)

    def read_action(self) -> dict[str, float]:
        return self._capturing.get_captured() if self._capturing else {}

    def get_possible_actions(self) -> list[str]:
        return self._controller.get_possible_actions() if self._controller else []
