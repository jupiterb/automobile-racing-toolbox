from pynput.keyboard import Listener, Key, Controller

from enviroments.real.interface.abstract import RealGameInterface
from schemas import GameGlobalConfiguration, GameSystemConfiguration, Action, State
from enviroments.real.capturing import ScreenCapturing, KeyboardCapturing
from enviroments.real.state import RealStateBuilder
import numpy as np

from schemas.enviroment.steering import SteeringAction 

Frame = np.ndarray 

class LocalInterface(RealGameInterface):
    def __init__(
        self,
        global_configuration: GameGlobalConfiguration,
        system_configuration: GameSystemConfiguration
    ) -> None:
        super().__init__(global_configuration, system_configuration)
        self._state_builder = RealStateBuilder(global_configuration)
        self._screen_capturing: ScreenCapturing = ScreenCapturing(
            global_configuration.process_name, 
            system_configuration.specified_window_rect
        )
        self._keyboard_capturing: KeyboardCapturing = KeyboardCapturing(
            set(global_configuration.action_key_mapping.values())
        )
        self._keayboard = Controller()

    def run(self) -> None:
        super().run()

    def reset(self) -> State:
        self._keyboard_capturing.reset()
        return super().reset()

    def read_state(self) -> State:
        driving_screenshot = self._screen_capturing.grab_image(
            self._system_configuration.driving_screen_frame
        )
        velocity_screenshot = self._screen_capturing.grab_image(
            self._system_configuration.velocity_screen_frame
        )
        self._state_builder.add_features_from_screenshot(driving_screenshot)
        self._state_builder.add_velocity_with_ocr(velocity_screenshot)
        return self._state_builder.build()

    def apply_keyboard_action(self, action: list[SteeringAction]) -> None:
        for a in action:
            self._keayboard.press(self._global_configuration.action_key_mapping[a])
        for a in set(SteeringAction) - set(action):
            self._keayboard.release(self._global_configuration.action_key_mapping[a])
        

    def read_action(self) -> Action:
        return Action(keys=self._keyboard_capturing.get_captured_keys())
