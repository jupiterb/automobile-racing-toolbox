import gym
import wandb
import numpy as np
from stable_baselines3 import DQN
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from conf.example_configuration import get_game_config
from interface.training_local import TrainingLocalGameInterface
from rl.enviroment import RealTimeEnviroment
from rl.final_state.detector import FinalStateDetector
from rl.config import FinalValueDetectionParameters, RewardConfig, ObservationConfig
from rl.builder import reward_wrappers, observation_wrappers


def main():
    config = {
        "policy": "CnnPolicy",
        "total_timesteps": 500_000,
        "buffer_size": 100_000,
        "learning_starts": 10_00,
        "gamma": 0.96,
    }

    run = wandb.init(
        project="test-sb3",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=False,
        entity="automobile-racing-toolbox",
        config=config,
    )

    env = DummyVecEnv([setup_env])

    model = DQN(
        env=env,
        policy=config["policy"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
    )
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    run.finish()


def setup_env() -> gym.Env:
    config = get_game_config()
    interface = TrainingLocalGameInterface(config)
    final_st_det = FinalStateDetector(
        [
            FinalValueDetectionParameters(
                feature_name="speed",
                min_value=2,
                max_value=float("inf"),
                required_repetitions_in_row=20,
                not_final_value_required=True,
            )
        ]
    )

    reward_conf = RewardConfig(
        speed_diff_thresh=15,
        memory_length=1,
        speed_diff_trans=np.abs,
        off_track_reward_trans=lambda reward: -abs(reward) - 100,
        clip_range=(-300, 300),
        baseline=100,
        scale=100,
    )

    observation_conf = ObservationConfig(
        shape=(50, 100), stack_size=4, lidar_config=None
    )

    env = RealTimeEnviroment(interface, final_st_det)
    env = reward_wrappers(env, reward_conf)
    env = observation_wrappers(env, observation_conf)
    return env


def debug():
    env = setup_env()
    env.reset()
    for _ in range(10000):
        _ = env.step(-1)


if __name__ == "__main__":
    # main()
    debug()
