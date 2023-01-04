import argparse
import json
from racing_toolbox.environment.config import (
    ActionConfig,
    RewardConfig,
    ObservationConfig,
    EnvConfig,
)
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.observation.config.lidar_config import LidarConfig
from racing_toolbox.observation.config.track_segmentation_config import (
    TrackSegmentationConfig,
)
from racing_toolbox.observation.utils import ScreenFrame
from racing_toolbox.training.worker.worker import Worker, Address


def main():
    args = get_cli_args()
    game_config = get_game_config(args.game_config)
    env_config = get_env_config()
    client = Worker(
        policy_address=Address(args.host, args.port),
        game_conf=game_config,
        env_conf=env_config,
    )
    wait_until_user_confirm_start()
    client.run()


def wait_until_user_confirm_start():
    print("Type `r` to run this trainee")
    while True:
        if input() == "r":
            break


def get_game_config(config_path):
    with open(config_path) as gp:
        return GameConfiguration(**json.load(gp))


def get_env_config() -> EnvConfig:
    action_config = ActionConfig(
        available_actions={
            "FORWARD": {0, 1, 2},
            "BREAK": set(),
            "RIGHT": {1, 3},
            "LEFT": {2, 4},
        }
    )

    action_config.available_actions = None

    reward_conf = RewardConfig(
        speed_diff_thresh=3,
        memory_length=2,
        speed_diff_trans=lambda x: float(x) ** 1.2,
        off_track_reward_trans=lambda reward: -abs(reward) - 100,
        clip_range=(-300, 300),
        baseline=20,
        scale=300,
    )

    lidar_config = LidarConfig(
        depth=3,
        angles_range=(-90, 90, 10),
        lidar_start=(0.9, 0.5),
    )
    track_config = TrackSegmentationConfig(
        track_color=(200, 200, 200),
        tolerance=80,
        noise_reduction=5,
    )

    observation_conf = ObservationConfig(
        frame=ScreenFrame(top=0.475, bottom=0.9125, left=0.01, right=0.99),
        shape=(60, 60),
        stack_size=4,
        lidar_config=None,
        track_segmentation_config=None,
    )

    return EnvConfig(
        action_config=action_config,
        reward_config=reward_conf,
        observation_config=observation_conf,
        max_episode_length=1_000,
    )


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, help="ip address of the policy server")
    parser.add_argument(
        "--port", type=int, default=9900, help="The port to use (on localhost)."
    )
    parser.add_argument(
        "--game_config",
        type=str,
        default="./config/trackmania/game_config.json",
        help="Path to json with game configuartion",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
