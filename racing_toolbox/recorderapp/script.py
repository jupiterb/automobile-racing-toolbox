import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from interface import LocalGameInterface
from interface.models import ControllerType
from recorderapp import EpisodeRecordingManager
from conf import get_game_config


def record(
    user_name: str, recording_name: str, controller_type: ControllerType
) -> None:
    recording_manager = EpisodeRecordingManager()
    recording_manager.start(
        LocalGameInterface(get_game_config(), controller_type),
        user_name,
        recording_name,
        get_game_config().frequency_per_second,
    )
    while True:
        if input() == "stop":
            recording_manager.release()
            return


if __name__ == "__main__":
    user_name = sys.argv[1]
    recording_name = sys.argv[2]
    controller = sys.argv[3]
    controller_type = (
        ControllerType.KEYBOARD
        if controller == "keyboard"
        else ControllerType.GAMEPAD
        if controller == "gamepad"
        else None
    )
    if controller_type is not None:
        record(user_name, recording_name, controller_type)
