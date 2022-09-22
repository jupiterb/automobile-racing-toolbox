from vgamepad import XUSB_BUTTON
from enum import Enum


max_xusb = max(XUSB_BUTTON)


class GamepadControl(Enum):
    LEFT_TRIGGER = max_xusb + 1
    RIGHT_TRIGGER = max_xusb + 2
    LEFT_JOYSTICK_X = max_xusb + 3
    LEFT_JOYSTICK_Y = max_xusb + 4
    RIGHT_JOYSTICK_X = max_xusb + 5
    RIGHT_JOYSTICK_Y = max_xusb + 6


GamepadAction = GamepadControl | XUSB_BUTTON
