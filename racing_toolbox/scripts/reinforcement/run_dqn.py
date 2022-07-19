from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from wandb.integration.sb3 import WandbCallback
import wandb

from conf.example_configuration import get_game_config
from interface.training_local import TrainingLocalGameInterface
from rl.wrappers.reward import OffTrackPunishment, SpeedDropPunishment, ClipReward
from rl.enviroment import RealTimeEnviroment
from rl.final_state.detector import FinalStateDetector
from rl.models.final_value_detecion_params import FinalValueDetectionParameters
from rl.wrappers.observation_squeezing import SqueezingWrapper


def main():
    config = {
        "policy": "CnnPolicy",
        "total_timesteps": 1_000_000, 
        "buffer_size": 100_000,
        "learning_starts": 1000,
    }

    run = wandb.init(
        project="test-sb3",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        entity="automobile-racing-toolbox",
        config=config,
    )

    env = DummyVecEnv([setup_env])
    # env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)

    model = DQN(
        env=env, 
        policy=config["policy"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        verbose=1, 
        tensorboard_log=f"runs/{run.id}"
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




def setup_env() -> RealTimeEnviroment:
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
    env = SpeedDropPunishment(env, 1, 15)
    env = OffTrackPunishment(env)
    env = ClipReward(env, -300, 300)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (50, 100))
    env = FrameStack(env, 4)
    env = SqueezingWrapper(env)
    return env 


def debug():
    env = setup_env()
    env.reset()
    for _ in range(10000):
        _ = env.step(-1)


if __name__ == "__main__":
    main()
    # debug()
