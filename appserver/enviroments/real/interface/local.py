from pynput.keyboard import Controller, Key
import numpy as np

from enviroments.real.interface.abstract import RealGameInterface
from schemas import GameGlobalConfiguration, GameSystemConfiguration, Action
from enviroments.real.capturing import ScreenCapturing, KeyboardCapturing
from enviroments.real.state.ocr import AbstractOcr, SegmentDetectionOcr
from schemas.game.feature_extraction import SegmentDetectionParams, OcrType
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
            global_configuration.apply_grayscale,
            global_configuration.window_size,
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
        self._keyboard = Controller()

    def run(self) -> None:
        super().run()

    def reset(self):
        self._keyboard_capturing.reset()

    def get_image_input(self) -> np.ndarray:
        image = self._screen_capturing.grab_image(
            self._system_configuration.driving_screen_frame
        )
        return image

    def get_velocity_input(self) -> int:
        velocity_screenshot = self._screen_capturing.grab_image(
            self._system_configuration.velocity_screen_frame
        )
        return self._ocr.read_number(velocity_screenshot)

    def apply_keyboard_action(self, action: list[SteeringAction]) -> None:
        for a in action:
            self._keyboard.press(self._global_configuration.action_key_mapping[a])
        for a in set(SteeringAction) - set(action):
            self._keyboard.release(self._global_configuration.action_key_mapping[a])

    def read_action(self) -> Action:
        return Action(
            keys={key.name for key in self._keyboard_capturing.get_captured_keys()}
        )
