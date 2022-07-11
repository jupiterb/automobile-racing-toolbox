from pynput.keyboard import Key
from  interface.models import GameConfiguration, SteeringAction, ScreenFrame, OcrConfiguration


def get_trackmania() -> GameConfiguration:
    return GameConfiguration(
        game_id="trackmania",
        process_name="Trackmania Nations Forever",
        window_size=(1000, 800),
        obervation_frame=ScreenFrame(),
        discrete_actions_mapping={
            SteeringAction.FORWARD: Key.up,
            SteeringAction.BREAK: Key.down,
            SteeringAction.RIGHT: Key.right,
            SteeringAction.LEFT: Key.left
        },
        ocrs={
            "speed" : (
                ScreenFrame(
                    top = 0.945,
                    bottom = 0.9875,
                    left = 0.918,
                    right = 0.9825
                ), 
                OcrConfiguration(
                    threshold = 235,
                    max_digits = 3,
                    segemnts_definitions = {
                        0: ScreenFrame(top=0, bottom=0.09, left=0.42, right=0.60),
                        1: ScreenFrame(top=0.15, bottom=0.28, left=0.14, right=0.28),
                        2: ScreenFrame(top=0.15, bottom=0.28, left=0.85, right=1.),
                        3: ScreenFrame(top=0.38, bottom=0.5, left=0.42, right=0.60),
                        4: ScreenFrame(top=0.58, bottom=0.73, left=0.14, right=0.28),
                        5: ScreenFrame(top=0.58, bottom=0.73, left=0.85, right=1.),
                        6: ScreenFrame(top=0.82, bottom=0.94, left=0.42, right=0.60),
                    }
                )
            )
        },
    )
    