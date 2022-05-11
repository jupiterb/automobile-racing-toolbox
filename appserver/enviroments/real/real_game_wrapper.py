from enviroments.common import RealTimeWrapper
from schemas import GameGlobalConfiguration, GameSystemConfiguration, Action, State
from enviroments.real.interface import ScreenCapturing, KeyboardCapturing
from enviroments.real.state import RealStateBuilder


class RealGameWrapper(RealTimeWrapper):

    def __init__(self, 
        global_configuration: GameGlobalConfiguration,
        system_configuration: GameSystemConfiguration
    ) -> None:
        super().__init__(global_configuration, system_configuration)
        self._screen_capturing: ScreenCapturing = ScreenCapturing(global_configuration.process_name, (0, 0, 1920, 1080))
        self._keyboard_capturing: KeyboardCapturing = KeyboardCapturing(global_configuration.control_actions)
        self._state_builder = RealStateBuilder(global_configuration)

    def run(self):
        super().run()

    def step(self, action: Action) -> State:
        return super().step(action)

    def reset(self) -> State:
        self._keyboard_capturing.reset()
        return super().reset()

    def read_state(self) -> State:
        driving_screenshot = self._screen_capturing.grab_image(self._system_configuration.driving_screen_frame)
        velocity_screenshot = self._screen_capturing.grab_image(self._system_configuration.velocity_screen_frame)
        self._state_builder.reset()
        self._state_builder.add_features_from_screenshot(driving_screenshot)
        self._state_builder.add_velocity_with_ocr(velocity_screenshot)
        return self._state_builder.get_result()

    def apply_action(self, action: Action):
        return super().apply_action(action)

    def read_action(self) -> Action:
        return Action(keys=self._keyboard_capturing.get_captured_keys())
