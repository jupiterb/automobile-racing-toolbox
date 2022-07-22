from pynput.keyboard import Controller, Key


class KeyboardController:
    def __init__(self, available_keys: set[Key]) -> None:
        self._available_keys = available_keys
        self._controller = Controller()

    def set_pressed(self, keys: list[Key]) -> None:
        for key in keys:
            if key in self._available_keys:
                self._controller.press(key)
        for key in set(self._available_keys) - set(keys):
            self._controller.release(key)

    def click(self, key: Key) -> None:
        self._controller.press(key)
        self._controller.release(key)
        