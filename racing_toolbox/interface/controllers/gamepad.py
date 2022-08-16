import vgamepad as vg

from interface.controllers.abstract import GameActionController
from interface.models import SteeringAction, GamepadAction
from interface.models.gamepad_action import GamepadControl


class GamepadController(GameActionController):
    def __init__(
        self,
        gamepad_action_mapping: dict[SteeringAction, GamepadAction],
        reset_sequence: list[GamepadAction],
    ) -> None:
        self._gamepad_action_mapping = gamepad_action_mapping
        self._reset_sequence = reset_sequence
        self._gamepad = vg.VX360Gamepad()

    def reset_game(self) -> None:
        gamepad_actions = {action: 1.0 for action in self._reset_sequence}
        self._apply_gamepad_actions(gamepad_actions)

    def apply_actions(self, actions: dict[SteeringAction, float]) -> None:
        gamepad_actions = {
            self._gamepad_action_mapping[action]: value
            for action, value in actions.items()
        }
        self._apply_gamepad_actions(gamepad_actions)

    def _apply_gamepad_actions(
        self, gamepad_actions: dict[GamepadAction, float]
    ) -> None:
        discrete_actions = {
            action for action in gamepad_actions if isinstance(action, vg.XUSB_BUTTON)
        }
        continous_actions = {
            action: value
            for action, value in gamepad_actions.items()
            if isinstance(action, GamepadControl)
        }
        self._apply_gamepad_discrete_actions(discrete_actions)
        self._apply_gamepad_continous_actions(continous_actions)
        self._gamepad.update()

    def _apply_gamepad_discrete_actions(self, buttons: set[vg.XUSB_BUTTON]) -> None:
        for action in buttons:
            self._gamepad.press_button(button=action)
        for action in set(vg.XUSB_BUTTON) - buttons:
            self._gamepad.release_button(button=action)

    def _apply_gamepad_continous_actions(
        self, controls: dict[GamepadControl, float]
    ) -> None:
        self._apply_triggers_actions(controls)
        self._apply_joystick_actions(controls)

    def _apply_triggers_actions(self, controls: dict[GamepadControl, float]) -> None:
        triggers = [GamepadControl.LEFT_TRIGGER, GamepadControl.RIGHT_TRIGGER]
        actions = [
            lambda gamepad, value: gamepad.left_trigger_float(value_float=value),
            lambda gamepad, value: gamepad.right_trigger_float(value_float=value),
        ]
        for trigger, action in zip(triggers, actions):
            if trigger in controls:
                action(self._gamepad, controls[trigger])
        self._gamepad.update()

    def _apply_joystick_actions(self, controls: dict[GamepadControl, float]) -> None:
        joysticks_x = [GamepadControl.LEFT_JOYSTICK_X, GamepadControl.RIGHT_JOYSTICK_X]
        joysticks_y = [GamepadControl.LEFT_JOYSTICK_Y, GamepadControl.RIGHT_JOYSTICK_Y]
        actions = [
            lambda gp, x, y: gp.left_joystick_float(x_value_float=x, y_value_float=y),
            lambda gp, x, y: gp.right_joystick_float(x_value_float=x, y_value_float=y),
        ]
        for joystick_x, joystick_y, action in zip(joysticks_x, joysticks_y, actions):
            if joystick_x in controls and joystick_y in controls:
                action(self._gamepad, controls[joystick_x], controls[joystick_y])
