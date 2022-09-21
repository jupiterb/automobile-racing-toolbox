import sys
from racing_toolbox.interface import from_config
from racing_toolbox.interface.controllers import KeyboardController, GamepadController
from racing_toolbox.interface.capturing import KeyboardCapturing, GamepadCapturing
from racing_toolbox.recorderapp import EpisodeRecordingManager
from racing_toolbox.datatool.recordings import BinaryFileRecordingsService
from racing_toolbox.conf import get_game_config
import time


def starting(seconds: int) -> None:
    print('To finish recording, type "q" and press Enter')
    for i in range(seconds):
        print(f"Recordong start at {seconds - i}s")
        time.sleep(1)
    print("Started.")


def record(user_name: str, recording_name: str, controller_type: str) -> None:
    starting(10)

    if controller_type == "gamepad":
        interface = from_config(get_game_config(), GamepadController, GamepadCapturing)
    elif controller_type == "keyboard":
        interface = from_config(
            get_game_config(), KeyboardController, KeyboardCapturing
        )
    else:
        raise NotImplementedError(
            f"Cannot create game interface with {controller_type} controller"
        )

    recording_manager = EpisodeRecordingManager(
        BinaryFileRecordingsService("./recordings")
    )
    recording_manager.start(
        interface,
        user_name,
        recording_name,
        get_game_config().frequency_per_second,
    )
    while True:
        if input() == "q":
            recording_manager.release()
            print("Finished.")
            return


if __name__ == "__main__":
    user_name = sys.argv[1]
    recording_name = sys.argv[2]
    controller = sys.argv[3]
    record(user_name, recording_name, controller)
