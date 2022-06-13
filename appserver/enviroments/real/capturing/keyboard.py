from pynput.keyboard import Listener, Key


class KeyboardCapturing:
    def __init__(self, available_keys_names: set[Key]) -> None:
        self._available_keys: set[Key] = available_keys_names
        self._last_keys: set[Key] = set()
        self._keyboard_listener: Listener = Listener(
            on_press=self._callback_on_press, on_release=self._callback_on_release
        )

    def reset(self):
        self._last_keys = set()
        try:
            self._keyboard_listener.join()
        except:
            pass
        self._keyboard_listener.start()

    def get_captured_keys(self) -> set[Key]:
        return self._last_keys.copy()

    def _callback_on_press(self, key):
        try:
            if key in self._available_keys:
                self._last_keys.add(key)
        except AttributeError:
            pass

    def _callback_on_release(self, key):
        try:
            if key in self._available_keys:
                self._last_keys.remove(key)
        except:
            pass
