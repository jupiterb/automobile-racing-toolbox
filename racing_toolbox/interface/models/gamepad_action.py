from vgamepad import XUSB_BUTTON
from enum import Enum


class GamepadTrigger(Enum):
    LEFT_TRIGGER = 0
    RIGHT_TRIGGER = 1
    LEFT_JOYSTICK_X = 2
    LEFT_JOYSTICK_Y = 3
    RIGHT_JOYSTICK_X = 4
    RIGHT_JOYSTICK_Y = 5


GamepadAction = GamepadTrigger | XUSB_BUTTON
