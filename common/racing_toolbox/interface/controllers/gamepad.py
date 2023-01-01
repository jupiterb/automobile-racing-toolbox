import vgamepad as vg
from typing import Optional

from racing_toolbox.interface.controllers.abstract import GameActionController
from racing_toolbox.interface.models import GamepadAction, GamepadControl, GamepadButton
from racing_toolbox.interface.models.gamepad_action import GamepadControl


class GamepadController(GameActionController[GamepadAction]):

    global_gamepad: Optional[vg.VX360Gamepad] = None

    def __init__(
        self,
        action_mapping: dict[str, GamepadAction],
        reset_sequence: list[GamepadAction],
    ) -> None:
        super().__init__(action_mapping=action_mapping, reset_sequence=reset_sequence)
        self._gamepad = (
            GamepadController.global_gamepad
            if GamepadController.global_gamepad
            else vg.VX360Gamepad()
        )

    def reset_game(self) -> None:
        gamepad_actions = {action: 1.0 for action in self._reset_sequence}
        self._apply_gamepad_actions(gamepad_actions)
        self._gamepad.reset()
        self._gamepad.update()

    def apply_actions(self, actions: dict[str, float]) -> None:
        gamepad_actions = {
            self._action_mapping[action]: value for action, value in actions.items()
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
        for axis, value in controls.items():
            self._apply_gamepad_axis_action(axis, value)

    def _apply_gamepad_axis_action(self, axis: GamepadControl, value: float) -> None:
        if axis == GamepadControl.AXIS_X_LEFT:
            y_value = GamepadController._normalize_axis(self._gamepad.report.sThumbLY)
            self._gamepad.left_joystick_float(value, y_value)
        elif axis == GamepadControl.AXIS_Y_LEFT:
            x_value = GamepadController._normalize_axis(self._gamepad.report.sThumbLX)
            self._gamepad.left_joystick_float(x_value, value)
        elif axis == GamepadControl.AXIS_X_RIGHT:
            y_value = GamepadController._normalize_axis(self._gamepad.report.sThumbRY)
            self._gamepad.right_joystick_float(value, y_value)
        elif axis == GamepadControl.AXIS_Y_RIGHT:
            x_value = GamepadController._normalize_axis(self._gamepad.report.sThumbRX)
            self._gamepad.right_joystick_float(x_value, value)
        elif axis == GamepadControl.AXIS_Z:
            right_value = value if value >= 0 else 0
            left_value = -value if value < 0 else 0
            self._gamepad.left_trigger_float(left_value)
            self._gamepad.right_trigger_float(right_value)

    @staticmethod
    def _normalize_axis(value: int) -> float:
        return value / 32767
