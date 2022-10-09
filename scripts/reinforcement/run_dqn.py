import gym
import wandb
from stable_baselines3 import DQN
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import TimeLimit

from racing_toolbox.conf.example_configuration import get_game_config
from racing_toolbox.interface.models.game_configuration import GameConfiguration
from racing_toolbox.environment.config.training import DQNConfig
from racing_toolbox.environment.wrappers.stats import WandbWrapper, LoggingWrapper
from racing_toolbox.environment.final_state.detector import FinalStateDetector
from racing_toolbox.environment.config import (
    FinalValueDetectionParameters,
    RewardConfig,
    ObservationConfig,
)
from racing_toolbox.environment.builder import reward_wrappers, observation_wrappers

import logging

logging.basicConfig(level=logging.INFO)

from racing_toolbox.interface import from_config
from racing_toolbox.interface.controllers import KeyboardController
from racing_toolbox.environment.wrappers import DiscreteActionToVectorWrapper


def get_configuration() -> tuple[
    GameConfiguration, ObservationConfig, RewardConfig, DQNConfig
]:
    game_conf = get_game_config()

    reward_conf = RewardConfig(
        speed_diff_thresh=3,
        memory_length=5,
        speed_diff_trans=lambda x: float(x) ** 1.3,
        off_track_reward_trans=lambda reward: -abs(reward) - 400,
        clip_range=(-400, 400),
        baseline=0,
        scale=400,
    )

    observation_conf = ObservationConfig(
        shape=(50, 100), stack_size=4, lidar_config=None, track_segmentation_config=None
    )

    train_conf = DQNConfig(
        policy="CnnPolicy",
        total_timesteps=500_000,
        buffer_size=500_000,
        learning_starts=50_00,
        gamma=0.99,
        exploration_final_epsilon=0.1,
        learning_rate=1e-4,
    )

    return game_conf, observation_conf, reward_conf, train_conf


def main():

    game_conf, obs_conf, rew_conf, train_conf = get_configuration()

    run = wandb.init(
        project="testsb3v3",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,
        entity="automobile-racing-toolbox",
        config={
            "training": dict(train_conf),
            "observation": dict(obs_conf),
            "reward": dict(rew_conf),
            # "game": dict(game_conf), # TODO: Add proper JSON encoder to the enums
        },
    )

    env = DummyVecEnv([lambda: setup_env(game_conf, rew_conf, obs_conf)])
    env = VecVideoRecorder(
        env,
        f"foo-videos/{run.id}",
        record_video_trigger=lambda x: x % 10_000 == 0,
        video_length=400,
    )

    model = DQN(
        env=env,
        policy=train_conf.policy,
        buffer_size=train_conf.buffer_size,
        learning_starts=train_conf.learning_starts,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        exploration_final_eps=train_conf.exploration_final_epsilon,
        learning_rate=train_conf.learning_rate,
        batch_size=256,
    )
    model.learn(
        total_timesteps=train_conf.total_timesteps,
        callback=WandbCallback(
            gradient_save_freq=10,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    run.finish()


def setup_env(
    config: GameConfiguration, reward_conf: RewardConfig, obs_conf: ObservationConfig
) -> gym.Env:
    interface = from_config(config, KeyboardController)

    final_st_det = FinalStateDetector(
        [
            FinalValueDetectionParameters(
                feature_name="speed",
                min_value=2,
                max_value=float("inf"),
                required_repetitions_in_row=20,
                not_final_value_required=False,
            )
        ]
    )

    env = gym.make(
        "custom/real-time-v0",
        game_interface=interface,
        final_state_detector=final_st_det,
    )

    available_actions = [
        {"FORWARD"},
        {"FORWARD", "LEFT"},
        {"FORWARD", "RIGHT"},
        {"LEFT"},
        {"RIGHT"},
        set(),
    ]

    env = DiscreteActionToVectorWrapper(
        env, available_actions, interface.get_possible_actions()
    )
    env = reward_wrappers(env, reward_conf)
    env = observation_wrappers(env, obs_conf)
    env = TimeLimit(env, 1_000)
    env = Monitor(env)
    env = WandbWrapper(env, 5)
    env = LoggingWrapper(env)

    print("PRESS S if your game is ready for training!")
    while True:
        if input() == "S":
            break

    return env


def debug():

    game_conf, obs_conf, rew_conf, train_conf = get_configuration()
    env = setup_env(game_conf, rew_conf, obs_conf)
    env.reset()
    for _ in range(10000):
        _, r, done, info = env.step(-1)
        # print(r)
        if done:
            env.reset()


if __name__ == "__main__":
    main()
    # debug()
