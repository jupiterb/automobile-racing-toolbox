from pynput.keyboard import Controller, Listener, Key


class Keyboard:
    def __init__(self, available_keys: set[Key]) -> None:
        self._available_keys = available_keys
        self._pressed: set[Key] = set()
        self._controller = Controller()
        self._listener: Listener = Listener(
            on_press=self._on_press, on_release=self._on_release
        )

    def reset(self):
        self._pressed = set()
        try:
            self._listener.join()
        except:
            pass
        self._listener.start()

    def set_pressed(self, keys: list[Key]) -> None:
        for key in keys:
            if key in self._available_keys:
                self._controller.press(key)
        for key in set(self._available_keys) - set(keys):
            self._controller.release(key)

    def get_pressed(self) -> list[Key]:
        return list(self._pressed)

    def _on_press(self, key):
        try:
            if key in self._available_keys:
                self._pressed.add(key)
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            if key in self._available_keys:
                self._pressed.remove(key)
        except:
            pass
