import argparse
from ray.rllib.env.policy_server_input import PolicyServerInput
import ray
import gym

from racing_toolbox.trainer import Trainer, config
from racing_toolbox.enviroment import builder
from racing_toolbox.enviroment.config.env import EnvConfig
from racing_toolbox.conf.example_configuration import get_game_config
from racing_toolbox.observation.config.lidar_config import LidarConfig
from racing_toolbox.observation.config.track_segmentation_config import (
    TrackSegmentationConfig,
)
from racing_toolbox.enviroment.config import (
    RewardConfig,
    ObservationConfig,
)

PORT = 8000
HOST = "0.0.0.0"
BATCH = 256


def main():
    args = get_cli_args()
    ray.init()

    def _input(ioctx):
        if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
            return PolicyServerInput(
                ioctx,
                args.host,
                args.port + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
            )
        else:
            return None

    game_config = get_game_config()
    env_config = get_env_config()
    algo_config = get_algorithm_config(args.run)

    model_config = config.ModelConfig(fcnet_hiddens=[512, 256], fcnet_activation="Relu")
    training_config = config.TrainingConfig(
        env=builder.setup_env(game_config, env_config),
        input=_input,
        num_workers=args.num_workers,
        rollout_fragment_length=BATCH // args.num_workers,
        train_batch_size=BATCH,
        max_iterations=args.stop_iters,
        algorithm=algo_config,
        model=model_config,
    )

    training = Trainer(training_config)
    training.run()


def get_env_config() -> EnvConfig:
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
        shape=(84, 84), stack_size=4, lidar_config=None, track_segmentation_config=None
    )

    return EnvConfig(
        reward_config=reward_conf,
        observation_config=observation_conf,
        max_episode_length=1_000,
    )


def get_algorithm_config(algo: str = "DQN"):
    if algo == "DQN":
        buffer_config = config.ReplayBufferConfig(capacity=50_000)
        return config.DQNConfig(
            v_min=-100, v_max=100, replay_buffer_config=buffer_config
        )
    else:
        raise NotImplementedError


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    # Example-specific args.
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help="The base-port to use. " f"Default is {PORT}.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=HOST,
        help="The host address ot run on. " f"Default is {HOST}.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="The number of workers to use. Each worker will create "
        "its own listening socket for incoming experiences.",
    )
    parser.add_argument(
        "--restore",
        type=str,
        default="",
        help="Restore algorithm state from given checkpoint",
    )

    # General args.
    parser.add_argument(
        "--run",
        default="DQN",
        choices=["DQN"],
        help="The RLlib-registered algorithm to use.",
    )
    parser.add_argument(
        "--stop-iters", type=int, default=200, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=50_000,
        help="Number of timesteps to train.",
    )
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=100.0,
        help="Reward at which we stop training.",
    )

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


if __name__ == "__main__":
    main()
