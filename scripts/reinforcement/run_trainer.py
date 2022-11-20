import argparse
from ray.rllib.env.policy_server_input import PolicyServerInput
import json
import pickle
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.observation.utils.screen_frame import ScreenFrame

from racing_toolbox.training import Trainer, config
from racing_toolbox.environment import builder
from racing_toolbox.observation.config.lidar_config import LidarConfig
from racing_toolbox.observation.config.track_segmentation_config import (
    TrackSegmentationConfig,
)
from racing_toolbox.environment.config import (
    RewardConfig,
    ObservationConfig,
    EnvConfig,
    ActionConfig,
)
from racing_toolbox.training.config.params import TrainingParams

PORT = 8000
HOST = "0.0.0.0"
BATCH = 256


def main():
    args = get_cli_args()

    def input_(ioctx):
        if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
            return PolicyServerInput(
                ioctx,
                args.host,
                args.port + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
            )
        else:
            return None

    game_config = get_game_config(args.game_config)
    env_config = get_env_config()
    algo_config = get_algorithm_config(args.run)
    model_config = config.ModelConfig(
        fcnet_hiddens=[100, 256],
        fcnet_activation="relu",
        conv_filters=[
            (32, (8, 8), 4),
            (64, (4, 4), 2),
            (64, (3, 3), 1),
            (64, (8, 8), 1),
        ],
    )
    training_config = config.TrainingConfig(
        num_rollout_workers=args.num_workers,
        rollout_fragment_length=BATCH // args.num_workers,
        train_batch_size=BATCH,
        max_iterations=args.stop_iters,
        algorithm=algo_config,
        model=model_config,
    )

    env = builder.setup_env(game_config, env_config)
    trainer_params = TrainingParams(
        **training_config.dict(),
        observation_space=env.observation_space,
        action_space=env.action_space,
        input_=input_,
    )

    weights = None
    if args.checkpoint_path is not None:
        with open(args.checkpoint_path, "rb") as f:
            model = pickle.load(f)

        value = model["worker"]
        weights = pickle.loads(value)["state"]["default_policy"]["weights"]

    training = Trainer(
        trainer_params, checkpoint_path=args.restore, pre_trained_weights=weights
    )
    training.run()


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
        speed_diff_exponent=1.2,
        off_track_reward=-100,
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


def get_algorithm_config(algo: str = "DQN"):
    if algo == "DQN":
        buffer_config = config.ReplayBufferConfig(capacity=1_000)
        return config.DQNConfig(
            v_min=-100, v_max=100, replay_buffer_config=buffer_config
        )
    else:
        raise NotImplementedError


def get_game_config(config_path):
    with open(config_path) as gp:
        return GameConfiguration(**json.load(gp))


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
        default=None,
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
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        default=None,
        help="Path to checkpoint with pretrained weights. "
        "They will be used in initialization of the model if provided.",
    )
    parser.add_argument(
        "--game_config",
        type=str,
        default="./config/trackmania/game_config.json",
        help="Path to json with game configuartion.",
    )

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


if __name__ == "__main__":
    main()
