import gym
import wandb
from stable_baselines3 import SAC
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import TimeLimit

from racing_toolbox.conf.example_configuration import get_game_config
from racing_toolbox.interface import from_config
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.interface.controllers import GamepadController
from racing_toolbox.enviroment.config.training import DQNConfig
from racing_toolbox.enviroment.wrappers import SplitBySignActionWrapper
from racing_toolbox.enviroment.wrappers.stats import WandbWrapper
from racing_toolbox.enviroment.final_state.detector import FinalStateDetector
from racing_toolbox.enviroment.config import (
    FinalValueDetectionParameters,
    RewardConfig,
    ObservationConfig,
)
from racing_toolbox.enviroment.builder import reward_wrappers, observation_wrappers


def get_configuration() -> tuple[
    GameConfiguration, ObservationConfig, RewardConfig, SACConfig
]:
    game_conf = get_game_config()

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

    train_conf = DQNConfig(
        policy="CnnPolicy",
        total_timesteps=500_000,
        buffer_size=100_000,
        learning_starts=50_00,
        gamma=0.99,
        learning_rate=1e-5,
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

    model = SAC(
        env=env,
        policy=train_conf.policy,
        # buffer_size=train_conf.buffer_size,
        # learning_starts=train_conf.learning_starts,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        learning_rate=0.00005,
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
    interface = from_config(config, GamepadController)

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

    # FORWARD if first value of action is positive, else BREAK
    env = SplitBySignActionWrapper(env, 0)

    env = reward_wrappers(env, reward_conf)
    env = observation_wrappers(env, obs_conf)
    env = TimeLimit(env, 1_000)
    env = Monitor(env)
    env = WandbWrapper(env, 5)

    print("PRESS S if your game is ready for training!")
    while True:
        if input() == "S":
            break

    return env


def debug():
    game_conf, obs_conf, rew_conf, train_conf = get_configuration()
    env = setup_env(game_conf, rew_conf, obs_conf)
    env.reset()
    episode_len = 0
    for _ in range(10000):
        episode_len += 1
        obs, r, done, info = env.step(-1)
        if done:
            env.reset()
            print(f"episode length: {episode_len}")
            episode_len = 0
            print(info)


if __name__ == "__main__":
    main()
    # debug()
