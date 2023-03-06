from pynput.keyboard import Listener

from automobile_training.sim_interface.action_capturing.abstract import ActionCapturing


class KeyboardCapturing(ActionCapturing):
    def __init__(self, key_to_action_mapping: dict[str, str]) -> None:
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

    def get_actions(self) -> dict[str, float]:
        return {
            a: 1 if k in self._pressed else 0
            for k, a in self._key_to_action_mapping.items()
        }

    def _on_press(self, key):
        key = key.name
        if key in self._key_to_action_mapping:
            self._pressed.add(key)

    def _on_release(self, key):
        key = key.name
        if key in self._key_to_action_mapping:
            try:
                self._pressed.remove(key)
            except KeyError:
                pass
