from pynput.keyboard import Listener, Key


class KeyboardCapturing():

    def __init__(self, available_keys: set[str]) -> None:
        self._available_keys: set[str] = available_keys
        self._last_keys: set[str] = set()
        self._keyboard_listener: Listener = Listener(on_press=self._callback)

    def reset(self):
        self._last_keys = set()
        self._keyboard_listener.start()

    def get_captured_keys(self) -> set[str]:
        captured_keys = self._last_keys
        self._last_keys = set()
        return captured_keys

    def _callback(self, key):
        try:
            key_name = key.name
            if key_name in self._available_keys:
                self._last_keys.add(key_name)
        except AttributeError:
            pass 
