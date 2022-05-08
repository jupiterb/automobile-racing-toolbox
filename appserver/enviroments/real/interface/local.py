from pynput.keyboard import Listener, Key, Controller

from enviroments.real.interface.abstract import RealGameInterface
from schemas import GameGlobalConfiguration, GameSystemConfiguration, Action, State
from enviroments.real.capturing import ScreenCapturing
from enviroments.real.state import RealStateBuilder



class LocalInterface(RealGameInterface):
    def __init__(
        self,
        global_configuration: GameGlobalConfiguration,
        system_configuration: GameSystemConfiguration,
    ) -> None:
        super().__init__(global_configuration, system_configuration)
        self._available_keys: set[str] = self._global_configuration.control_actions
        self._screen_capturing: ScreenCapturing = ScreenCapturing(global_configuration.process_name)
        self._state_builder = RealStateBuilder()
        self._keyboard_listener = Listener(on_press=self._callback)
        self._last_keys: set[str] = set()
        self._keayboard = Controller()

    def run(self):
        super().run()

    def step(self, action: Action) -> State:
        return super().step(action)

    def reset(self) -> State:
        self._last_keys = set()
        self._keyboard_listener.start()
        return super().reset()

    def read_state(self) -> State:
        driving_screenshot = self._screen_capturing.grab_image(
            self._system_configuration.driving_screen_frame
        )
        velocity_screenshot = self._screen_capturing.grab_image(
            self._system_configuration.velocity_screen_frame
        )
        self._state_builder.reset()
        self._state_builder.add_features_from_screenshot(driving_screenshot)
        self._state_builder.add_velocity_with_ocr(velocity_screenshot)
        return self._state_builder.get_result()

    def apply_action(self, action: list[Key]):
        for key in action:
            self._keayboard.press(key)
        

    def read_action(self) -> Action:
        print(self._last_keys)
        action = Action(keys=self._last_keys)
        self._last_keys = set()
        return action

    def _callback(self, key):
        try:
            key_name = key.name
            if key_name in self._available_keys:
                self._last_keys.add(str(key_name))
        except AttributeError:
            pass
