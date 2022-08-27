import sys
from interface import GameInterfaceBuilder
from recorderapp import EpisodeRecordingManager
from conf import get_game_config
import time


def starting(seconds: int) -> None:
    print('To finish recording, type "q" and press Enter')
    for i in range(seconds):
        print(f"Recordong start at {seconds - i}s")
        time.sleep(1)
    print("Started.")


def record(user_name: str, recording_name: str, controller_type: str) -> None:
    starting(10)

    interface_builder = GameInterfaceBuilder()
    interface_builder.new_interface(get_game_config())
    if controller_type == "gamepad":
        interface_builder.with_gamepad_controller()
        interface_builder.with_gamepad_capturing()
    elif controller_type == "keyboard":
        interface_builder.with_keyborad_controller()
        interface_builder.with_keyboard_capturing()
    interface = interface_builder.build()

    recording_manager = EpisodeRecordingManager()
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
