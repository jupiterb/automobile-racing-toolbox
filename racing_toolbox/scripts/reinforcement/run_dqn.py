import gym
import wandb
import numpy as np
from stable_baselines3 import DQN
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import RecordEpisodeStatistics, RecordVideo, TimeLimit

from conf.example_configuration import get_game_config
from interface.training_local import TrainingLocalGameInterface
from rl.wrappers.stats import WandbWrapper
from rl.enviroment import RealTimeEnviroment
from rl.final_state.detector import FinalStateDetector
from rl.config import FinalValueDetectionParameters, RewardConfig, ObservationConfig
from rl.builder import reward_wrappers, observation_wrappers\


def main():
    config = {
        "policy": "CnnPolicy",
        "total_timesteps": 500_000, 
        "buffer_size": 100_000,
        "learning_starts": 50_00,
        "gamma": 0.99,
        "exploration_final_epsilon": 0.08
    }

    run = wandb.init(
        project="testsb3v3",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,
        entity="automobile-racing-toolbox",
        config=config,
    )

    env = DummyVecEnv([setup_env])
    env = VecVideoRecorder(env, f"foo-videos/{run.id}", record_video_trigger=lambda x: x % 10_000 == 0, video_length=400)
    
  
    model = DQN(
        env=env, 
        policy=config["policy"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        verbose=1, 
        tensorboard_log=f"runs/{run.id}",
        exploration_final_eps=config["exploration_final_epsilon"],
        learning_rate=0.00005
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
                not_final_value_required=False,
            )
        ]
    )

    reward_conf = RewardConfig(
        speed_diff_thresh=3,
        memory_length=1,
        speed_diff_trans=lambda x: float(x) ** 2,
        off_track_reward_trans=lambda reward: -abs(reward) - 400,
        clip_range=(-400, 400),
        baseline=0,
        scale=400
    )

    observation_conf = ObservationConfig(
        shape=(50, 100),
        stack_size=4
    )

    env = gym.make("custom/real-time-v0", game_interface=interface, final_state_detector=final_st_det)
    env = reward_wrappers(env, reward_conf)
    env = observation_wrappers(env, observation_conf)
    env = TimeLimit(env, 1_000)
    env = Monitor(env)
    env = WandbWrapper(env, 1)
    return env 


def debug():
    env = setup_env()
    env.reset()
    c = 0
    for _ in range(10000):
        c += 1
        _, r, done, info = env.step(-1)
        # print(f"rewrd {r}")
        if done:
            env.reset()
            print(c)
            c = 0
            print(info)


if __name__ == "__main__":
    main()
    # debug()
