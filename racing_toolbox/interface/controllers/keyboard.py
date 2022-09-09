from pynput.keyboard import Controller, Key
from interface.controllers.abstract import GameActionController


class KeyboardController(GameActionController):
    def __init__(self, action_to_key_mapping: dict[str, Key], reset_sequence: list[Key]) -> None:
        self._action_to_key_mapping = action_to_key_mapping
        self._reset_sequence = reset_sequence
        self._controller = Controller()

    def reset_game(self) -> None:
        for key in self._reset_sequence:
            self._controller.press(key)
            self._controller.release(key)

    def apply_actions(self, actions: dict[str, float]) -> None:
        actions_set = {
            action 
            for action, value in actions.items() 
            if action in self._action_to_key_mapping and value > 0
        }
        for action in actions_set:
            key = self._action_to_key_mapping[action]
            self._controller.press(key)
        for action in set(self._action_to_key_mapping) - actions_set:
            key = self._action_to_key_mapping[action]
            self._controller.release(key)

    def get_possible_actions(self) -> list[str]:
        return list(self._action_to_key_mapping)
        