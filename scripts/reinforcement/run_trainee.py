import argparse
from racing_toolbox.conf.example_configuration import get_game_config
from racing_toolbox.environment.config import (
    ActionConfig,
    RewardConfig,
    ObservationConfig,
    EnvConfig,
)
from racing_toolbox.observation.config.lidar_config import LidarConfig
from racing_toolbox.observation.config.track_segmentation_config import (
    TrackSegmentationConfig,
)
from racing_toolbox.observation.utils import ScreenFrame
from racing_toolbox.trainer.trainee import Trainee, Address


def main():
    args = get_cli_args()
    game_config = get_game_config()
    env_config = get_env_config()
    client = Trainee(
        policy_address=Address(args.host, args.port),
        game_conf=game_config,
        env_conf=env_config,
    )
    client.run()


def get_env_config() -> EnvConfig:
    action_config = ActionConfig(
        available_actions={
            "FORWARD": {0, 1, 2},
            "BREAK": set(),
            "RIGHT": {1, 3},
            "LEFT": {2, 4},
        }
    )

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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
