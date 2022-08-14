from pynput.keyboard import Listener, Key
from interface.models import SteeringAction
from interface.capturing.abstract import GameActionCapturing


class KeyboardCapturing(GameActionCapturing):
    def __init__(self, key_to_action_mapping: dict[Key, SteeringAction]) -> None:
        self._key_to_action_mapping = key_to_action_mapping
        self._pressed: set[Key] = set()
        self._listener: Listener = Listener(
            on_press=self._on_press, on_release=self._on_release
        )

    def stop(self):
        try:
            self._listener.join()
        except:
            pass
        self._pressed = set()

    def start(self):
        self._listener.start()

    def get_captured(self) -> dict[SteeringAction, float]:
        return {self._key_to_action_mapping[key]: 1 for key in self._pressed}

    def _on_press(self, key):
        try:
            if key in self._key_to_action_mapping:
                self._pressed.add(key)
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            if key in self._key_to_action_mapping:
                self._pressed.remove(key)
        except:
            pass
