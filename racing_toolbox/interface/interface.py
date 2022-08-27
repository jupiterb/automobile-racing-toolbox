import numpy as np
import time

from interface.models import SteeringAction
from interface.screen import ScreenProvider
from interface.capturing import GameActionCapturing
from interface.controllers import GameActionController
from interface.ocr import OcrWrapper


class GameInterface:
    def __init__(
        self, game_id: str, screen: ScreenProvider, reset_seconds: int
    ) -> None:
        self._name = game_id
        self._screen = screen
        self._reset_seconds = reset_seconds
        self._controller: GameActionController | None = None
        self._capturing: GameActionCapturing | None = None
        self._ocrs: list[OcrWrapper] = []

    def set_controller(self, controller: GameActionController | None) -> None:
        self._controller = controller

    def set_capturing(self, capturing: GameActionCapturing | None) -> None:
        self._capturing = capturing

    def set_ocrs(self, ocrs: list[OcrWrapper]) -> None:
        self._ocrs = ocrs

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

    def perform_ocr(self, on_last=True) -> dict[str, float]:
        return {
            ocr.name(): ocr.read_numer_from(self._screen, on_last) for ocr in self._ocrs
        }

    def apply_action(self, actions: dict[SteeringAction, float]) -> None:
        if self._controller:
            self._controller.apply_actions(actions)

    def read_action(self) -> dict[SteeringAction, float]:
        return self._capturing.get_captured() if self._capturing else {}
