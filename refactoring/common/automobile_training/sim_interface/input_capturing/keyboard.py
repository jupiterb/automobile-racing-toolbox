from pynput.keyboard import Listener

from automobile_training.sim_interface.input_capturing.abstract import InputCapturing


class KeyboardCapturing(InputCapturing):
    def __init__(self, key_to_input_mapping: dict[str, str]) -> None:
        self._key_to_input_mapping = key_to_input_mapping
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

    def get_inputs(self) -> dict[str, float]:
        return {
            input: 1 if key.name in self._pressed else 0
            for key, input in self._key_to_input_mapping.items()
        }

    def _on_press(self, key):
        if key in self._key_to_input_mapping:
            self._pressed.add(key.name)

    def _on_release(self, key):
        try:
            if key in self._key_to_input_mapping:
                self._pressed.remove(key.name)
        except KeyError:
            pass
