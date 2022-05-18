from pynput.keyboard import Listener, Key


class KeyboardCapturing():

    def __init__(self, available_keys: set[Key]) -> None:
        self._available_keys: set[Key] = available_keys
        self._last_keys: set[Key] = set()
        self._keyboard_listener: Listener = Listener(on_press=self._callback)

    def reset(self):
        self._last_keys = set()
        self._keyboard_listener.start()

    def get_captured_keys(self) -> set[Key]:
        captured_keys = self._last_keys
        self._last_keys = set()
        return captured_keys

    def _callback(self, key):
        try:
            if key in self._available_keys:
                self._last_keys.add(key)
        except AttributeError:
            pass 
