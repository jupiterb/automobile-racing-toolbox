from pynput.keyboard import Controller, Key
from racing_toolbox.interface.controllers.abstract import GameActionController


class KeyboardController(GameActionController[Key]):
    def __init__(
        self, action_mapping: dict[str, Key], reset_sequence: list[Key]
    ) -> None:
        super().__init__(action_mapping=action_mapping, reset_sequence=reset_sequence)
        self._controller = Controller()

    def reset_game(self) -> None:
        for key in self._reset_sequence:
            self._controller.press(key)
            self._controller.release(key)

    def apply_actions(self, actions: dict[str, float]) -> None:
        actions_set = {
            action
            for action, value in actions.items()
            if action in self._action_mapping and value > 0
        }
        for action in actions_set:
            key = self._action_mapping[action]
            self._controller.press(key)
        for action in set(self._action_mapping) - actions_set:
            key = self._action_mapping[action]
            self._controller.release(key)
