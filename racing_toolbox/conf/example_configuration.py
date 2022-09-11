from pynput.keyboard import Key
from vgamepad import XUSB_BUTTON
from interface.models import (
    GameConfiguration,
    ScreenFrame,
    OcrConfiguration,
)
from interface.models.gamepad_action import GamepadControl


def get_game_config() -> GameConfiguration:
    return GameConfiguration(
        game_id="trackmania",
        process_name="Trackmania Nations Forever",
        window_size=(1000, 800),
        obervation_frame=ScreenFrame(top=0.475, bottom=0.9125, left=0.01, right=0.99),
        discrete_actions_mapping={
            "FORWARD": Key.up,
            "BREAK": Key.down,
            "RIGHT": Key.right,
            "LEFT": Key.left,
        },
        continous_actions_mapping={
            "FORWARD": XUSB_BUTTON.XUSB_GAMEPAD_A,
            "BREAK": XUSB_BUTTON.XUSB_GAMEPAD_B,
            "DIRECT_X": GamepadControl.LEFT_JOYSTICK_X,
            "DIRECT_Y": GamepadControl.LEFT_JOYSTICK_Y,
        },
        reset_seconds=3,
        reset_keys_sequence=[Key.enter],
        reset_gamepad_sequence=[XUSB_BUTTON.XUSB_GAMEPAD_X],
        frequency_per_second=10,
        ocrs={
            "speed": (
                ScreenFrame(top=0.945, bottom=0.9875, left=0.918, right=0.9825),
                OcrConfiguration(
                    threshold=205,
                    max_digits=3,
                    segemnts_definitions={
                        0: ScreenFrame(top=0, bottom=0.09, left=0.42, right=0.60),
                        1: ScreenFrame(top=0.15, bottom=0.28, left=0.14, right=0.28),
                        2: ScreenFrame(top=0.15, bottom=0.28, left=0.85, right=1.0),
                        3: ScreenFrame(top=0.38, bottom=0.5, left=0.42, right=0.60),
                        4: ScreenFrame(top=0.58, bottom=0.73, left=0.14, right=0.28),
                        5: ScreenFrame(top=0.58, bottom=0.73, left=0.85, right=1.0),
                        6: ScreenFrame(top=0.82, bottom=0.94, left=0.42, right=0.60),
                    },
                ),
            )
        },
    )
