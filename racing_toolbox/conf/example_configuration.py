from pynput.keyboard import Key
from  interface.models import GameConfiguration, SteeringAction, ScreenFrame


def get_trackmania() -> GameConfiguration:
    return GameConfiguration(
        game_id="trackmania",
        process_name="Trackmania Nations Forever",
        window_size=(1000, 800),
        obervation_frame=ScreenFrame(),
        ocr_frames=[ScreenFrame()],
        discrete_actions_mapping={
            SteeringAction.FORWARD: Key.up,
            SteeringAction.BREAK: Key.down,
            SteeringAction.RIGHT: Key.right,
            SteeringAction.LEFT: Key.left
        }
    )
    