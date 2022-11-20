import argparse
import json
from racing_toolbox.interface import from_config
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.interface.controllers import KeyboardController, GamepadController
from racing_toolbox.interface.capturing import KeyboardCapturing, GamepadCapturing
from racing_toolbox.recorderapp import EpisodeRecordingManager
import time


def starting(seconds: int) -> None:
    print('To finish recording, type "q" and press Enter')
    for i in range(seconds):
        print(f"Recordong start at {seconds - i}s")
        time.sleep(1)
    print("Started.")


def record(game_config_path: str, user_name: str, recording_name: str, controller_type: str) -> None:
    game_config: GameConfiguration
    with open(game_config_path) as gp:
        game_config = GameConfiguration(**json.load(gp))

    starting(10)

    if controller_type == "gamepad":
        interface = from_config(game_config, GamepadController, GamepadCapturing)
    elif controller_type == "keyboard":
        interface = from_config(game_config, KeyboardController, KeyboardCapturing)
    else:
        raise NotImplementedError(
            f"Cannot create game interface with {controller_type} controller"
        )

    recording_manager = EpisodeRecordingManager()
    recording_manager.start(
        interface,
        user_name,
        recording_name,
        game_config.frequency_per_second,
    )
    while True:
        if input() == "q":
            recording_manager.release()
            print("Finished.")
            return


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game_config",
        type=str,
        default="./config/trackmania/game_config.json",
        help="Path to json with game configuartion",
    )
    parser.add_argument(
        "--user",
        type=str,
        help="User name",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of recording",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="keyboard"
        help="Controller type. Possible options are `keyboard` and `gamepad`",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_cli_args()
    record(args.game_config, args.user, args.name, args.controller)
