from pynput.keyboard import Key
from interface.models import (
    GameConfiguration,
    SteeringAction,
    ScreenFrame,
    OcrConfiguration,
)


def get_game_config() -> GameConfiguration:
    return GameConfiguration(
        game_id="trackmania",
        process_name="Trackmania Nations Forever",
        window_size=(1000, 800),
        obervation_frame=ScreenFrame(top=0.475, bottom=0.9125, left=0.01, right=0.99),
        discrete_actions_mapping={
            SteeringAction.FORWARD: Key.up,
            SteeringAction.BREAK: Key.down,
            SteeringAction.RIGHT: Key.right,
            SteeringAction.LEFT: Key.left,
        },
        reset_seconds=3,
        reset_keys_sequence=[Key.enter],
        frequency_per_second=10,
        ocrs={
            "speed": (
                ScreenFrame(top=0.94, bottom=0.99, left=0.91, right=0.99),
                OcrConfiguration(
                    threshold=190,
                    segments_coordinates={
                        0: (0.14, 0.5),
                        1: (0.34, 0.15),
                        2: (0.34, 0.5),
                        3: (0.34, 0.84),
                        4: (0.5, 0.5),
                        5: (0.66, 0.15),
                        6: (0.65, 0.5),
                        7: (0.65, 0.84),
                        8: (0.85, 0.5),
                    },
                ),
            )
        },
    )
