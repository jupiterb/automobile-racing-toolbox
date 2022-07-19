from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, Monitor, RecordEpisodeStatistics
from wandb.integration.sb3 import WandbCallback
import wandb
import numpy as np 

from conf.example_configuration import get_game_config
from interface.training_local import TrainingLocalGameInterface
from rl.wrappers.reward import OffTrackPunishment, SpeedDropPunishment, ClipReward, StandarizeReward
from rl.enviroment import RealTimeEnviroment
from rl.final_state.detector import FinalStateDetector
from rl.models.final_value_detecion_params import FinalValueDetectionParameters
from rl.wrappers.observation import SqueezingWrapper, RescaleWrapper

def main():
    config = {
        "policy": "CnnPolicy",
        "total_timesteps": 25_000, 
        "buffer_size": 100_000,
        "learning_starts": 1000,
        "gamma": 0.96,
        "learni"
    }

    run = wandb.init(
        project="test-sb3v2",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        entity="automobile-racing-toolbox",
        config=config,
    )

    env = DummyVecEnv([lambda: setup_env(run)])

    model = DQN(
        env=env, 
        policy=config["policy"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        verbose=1, 
        tensorboard_log=f"runs/{run.id}",
        gamma=config["gamma"],
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


def setup_env(run=None) -> RealTimeEnviroment:
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
    env = RealTimeEnviroment(interface, final_st_det)
    if run:
        env = Monitor(env, f"videos_{run.id}")      # record videos
    env = reward_wrappers(env)
    env = observations_wrappers(env)
    env = RecordEpisodeStatistics(env)   # record videos
    return env 


def observations_wrappers(env):
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (50, 100))
    env = RescaleWrapper(env)
    env = FrameStack(env, 4)
    env = SqueezingWrapper(env)
    return env 


def reward_wrappers(env):
    env = SpeedDropPunishment(env, 1, 1, np.abs)
    env = OffTrackPunishment(env)
    env = ClipReward(env, -400, 400)
    env = StandarizeReward(env, 100, 100)
    return env 


def debug():
    env = setup_env()
    env.reset()
    for _ in range(10000):
        _, reward, _, _ = env.step(-1)
        print(reward)

if __name__ == "__main__":
    main()
