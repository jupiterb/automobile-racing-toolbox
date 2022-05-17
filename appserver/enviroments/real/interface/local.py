from pynput.keyboard import Listener, Key, Controller

from enviroments.real.interface.abstract import RealGameInterface
from schemas import GameGlobalConfiguration, GameSystemConfiguration, Action, State
from enviroments.real.capturing import ScreenCapturing
from enviroments.real.state import RealStateBuilder
import numpy as np

from schemas.enviroment.steering import SteeringAction 

Frame = np.ndarray 

class LocalInterface(RealGameInterface):
    def __init__(
        self,
        global_configuration: GameGlobalConfiguration,
        system_configuration: GameSystemConfiguration,
    ) -> None:
        super().__init__(global_configuration, system_configuration)
        self._available_keys: set[SteeringAction] = set(self._global_configuration.action_key_mapping.keys())
        self._screen_capturing: ScreenCapturing = ScreenCapturing(global_configuration.process_name)
        self._keyboard_listener = Listener(on_press=self._callback)
        self._last_keys: set[str] = set()
        self._keayboard = Controller()

    def run(self) -> None:
        super().run()

    def reset(self) -> State:
        self._last_keys = set()
        # self._keyboard_listener.start()
        return super().reset()

    def read_frame(self) -> Frame:
        driving_screenshot = self._screen_capturing.grab_image(
            self._system_configuration.driving_screen_frame
        )
        velocity_screenshot = self._screen_capturing.grab_image(
            self._system_configuration.velocity_screen_frame
        )
        return driving_screenshot

    def apply_keyboard_action(self, action: list[SteeringAction]) -> None:
        for a in action:
            self._keayboard.press(self._global_configuration.action_key_mapping[a])
        for a in set(SteeringAction) - set(action):
            self._keayboard.release(self._global_configuration.action_key_mapping[a])
        

    def read_action(self) -> Action:
        print(self._last_keys)
        action = Action(keys=self._last_keys)
        self._last_keys = set()
        return action

    def _callback(self, key) -> None:
        try:
            if key in self._available_keys:
                self._last_keys.add(str(key))
        except AttributeError:
            pass
