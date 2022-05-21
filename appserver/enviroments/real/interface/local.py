from pynput.keyboard import Controller

from enviroments.real.interface.abstract import RealGameInterface
from schemas import GameGlobalConfiguration, GameSystemConfiguration, Action, State
from enviroments.real.capturing import ScreenCapturing, KeyboardCapturing
from enviroments.real.state.ocr import AbstractOcr, SegmentDetectionOcr
from schemas.game.feature_extraction import SegmentDetectionParams, OcrType

import numpy as np
import cv2

from schemas.enviroment.steering import SteeringAction

Frame = np.ndarray


class LocalInterface(RealGameInterface):
    def __init__(
        self,
        global_configuration: GameGlobalConfiguration,
        system_configuration: GameSystemConfiguration,
    ) -> None:
        super().__init__(global_configuration, system_configuration)
        self._screen_capturing: ScreenCapturing = ScreenCapturing(
            global_configuration.process_name,
            system_configuration.specified_window_rect,
        )
        self._keyboard_capturing: KeyboardCapturing = KeyboardCapturing(
            set(global_configuration.action_key_mapping.values())
        )
        ocr_velocity_params = global_configuration.ocr_velocity_params
        if (
            ocr_velocity_params.ocr_type is OcrType.SEGMENT_DETECTION
            and ocr_velocity_params.segment_detection_params
        ):
            self._ocr: AbstractOcr = SegmentDetectionOcr(
                global_configuration, ocr_velocity_params.segment_detection_params
            )
        else:
            self._ocr: AbstractOcr = SegmentDetectionOcr(
                global_configuration, SegmentDetectionParams()
            )
        self._keayboard = Controller()

    def run(self) -> None:
        super().run()

    def reset(self):
        self._keyboard_capturing.reset()

    def get_image_input(self) -> np.ndarray:
        image = self._screen_capturing.grab_image(
            self._system_configuration.driving_screen_frame
        )
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def get_velocity_input(self) -> int:
        velocity_screenshot = self._screen_capturing.grab_image(
            self._system_configuration.velocity_screen_frame
        )
        return self._ocr.read_number(velocity_screenshot)

    def apply_keyboard_action(self, action: list[SteeringAction]) -> None:
        for a in action:
            self._keayboard.press(self._global_configuration.action_key_mapping[a])
        for a in set(SteeringAction) - set(action):
            self._keayboard.release(self._global_configuration.action_key_mapping[a])

    def read_action(self) -> Action:
        return Action(
            keys=set([key.name for key in self._keyboard_capturing.get_captured_keys()])
        )
