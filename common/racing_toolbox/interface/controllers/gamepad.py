import vgamepad as vg

from racing_toolbox.interface.controllers.abstract import GameActionController
from racing_toolbox.interface.models import GamepadAction, GamepadControl, GamepadButton
from racing_toolbox.interface.models.gamepad_action import GamepadControl


class GamepadController(GameActionController[GamepadAction]):
    def __init__(
        self,
        action_mapping: dict[str, GamepadAction],
        reset_sequence: list[GamepadAction],
    ) -> None:
        super().__init__(action_mapping=action_mapping, reset_sequence=reset_sequence)
        self._gamepad = vg.VX360Gamepad()

    def reset_game(self) -> None:
        gamepad_actions = {action: 1.0 for action in self._reset_sequence}
        self._apply_gamepad_actions(gamepad_actions)
        self._gamepad.reset()
        self._gamepad.update()

    def apply_actions(self, actions: dict[str, float]) -> None:
        gamepad_actions = {
            self._action_mapping[action]: value
            for action, value in actions.items()
        }
        self._apply_gamepad_actions(gamepad_actions)

    def _apply_gamepad_actions(
        self, gamepad_actions: dict[GamepadAction, float]
    ) -> None:
        discrete_actions = {
            action
            for action, value in gamepad_actions.items()
            if isinstance(action, GamepadButton) and value > 0
        }
        continous_actions = {
            action: value
            for action, value in gamepad_actions.items()
            if isinstance(action, GamepadControl)
        }
        self._apply_gamepad_discrete_actions(discrete_actions)
        self._apply_gamepad_continous_actions(continous_actions)
        self._gamepad.update()

    def _apply_gamepad_discrete_actions(self, buttons: set[GamepadButton]) -> None:
        vg_buttons = {vg.XUSB_BUTTON[b.value] for b in buttons} 
        for action in vg_buttons:
            self._gamepad.press_button(button=action)
        for action in set(vg.XUSB_BUTTON) - vg_buttons:
            self._gamepad.release_button(button=action)

    def _apply_gamepad_continous_actions(
        self, controls: dict[GamepadControl, float]
    ) -> None:
        self._apply_triggers_actions(controls)
        self._apply_joystick_actions(controls)

    def _apply_triggers_actions(self, controls: dict[GamepadControl, float]) -> None:
        triggers = [GamepadControl.LEFT_TRIGGER, GamepadControl.RIGHT_TRIGGER]
        actions = [
            self._gamepad.left_trigger_float,
            self._gamepad.right_trigger_float,
        ]
        for trigger, action in zip(triggers, actions):
            if trigger in controls:
                action(controls[trigger])

    def _apply_joystick_actions(self, controls: dict[GamepadControl, float]) -> None:
        joysticks_x = [GamepadControl.LEFT_JOYSTICK_X, GamepadControl.RIGHT_JOYSTICK_X]
        joysticks_y = [GamepadControl.LEFT_JOYSTICK_Y, GamepadControl.RIGHT_JOYSTICK_Y]
        actions = [
            self._gamepad.left_joystick_float,
            self._gamepad.right_joystick_float,
        ]
        for joystick_x, joystick_y, action in zip(joysticks_x, joysticks_y, actions):
            if joystick_x in controls and joystick_y in controls:
                action(controls[joystick_x], controls[joystick_y])