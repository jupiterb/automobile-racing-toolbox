from pynput.keyboard import Listener, Key
from common.racing_toolbox.interface.models.keyboard_action import KeyAction
from racing_toolbox.interface.capturing.abstract import GameActionCapturing


class KeyboardCapturing(GameActionCapturing):
    def __init__(self, key_to_action_mapping: dict[KeyAction, str]) -> None:
        self._key_to_action_mapping = key_to_action_mapping
        self._pressed: set[str] = set()
        self._listener: Listener = Listener(
            on_press=self._on_press, on_release=self._on_release
        )

    def stop(self):
        try:
            self._listener.stop()
        except RuntimeError:
            pass
        self._pressed = set()

    def start(self):
        self._listener.start()

    def get_captured(self) -> dict[str, float]:
        return {
            action: 1 if key.name in self._pressed else 0
            for key, action in self._key_to_action_mapping.items()
        }

    def _on_press(self, key):
        if key in self._key_to_action_mapping:
            self._pressed.add(key.name)

    def _on_release(self, key):
        try:
            if key in self._key_to_action_mapping:
                self._pressed.remove(key.name)
        except KeyError:
            pass
