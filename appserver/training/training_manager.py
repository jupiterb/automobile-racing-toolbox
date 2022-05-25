from schemas import (
    GameSystemConfiguration,
    GameGlobalConfiguration,
    TrainingResult,
    Training,
)
from enviroments.real.env import RealTimeEnv
from enviroments.real.interface.local import LocalInterface
from schemas.game.game_global_configuration import GameGlobalConfiguration
from schemas.game.game_system_configuration import GameSystemConfiguration
from training.callback import SaveOnBestTrainingRewardCallback

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import CnnPolicy

import gym, os


class TrainingManager:
    BEST_MODEL_NAME = "best_model"
    LATEST_MODEL_NAME = "latest_model"

    def run_training(
        self,
        system_configuration: GameSystemConfiguration,
        global_configuration: GameGlobalConfiguration,
        training: Training,
    ):
        os.makedirs(training.parameters.tensorboard_dir_path, exist_ok=True)
        os.makedirs(training.parameters.log_dir_path, exist_ok=True)

        interface = LocalInterface(global_configuration, system_configuration)
        env = RealTimeEnv(interface)
        self._run_new_training(env, training)

    @staticmethod
    def _run_new_training(env: gym.Env, training: Training):
        model = DQN(
            CnnPolicy,
            env,
            verbose=1,
            optimize_memory_usage=False,
            buffer_size=10_000,
            tensorboard_log=training.parameters.tensorboard_dir_path,
        )

        callback = SaveOnBestTrainingRewardCallback(
            2000, training.parameters.log_dir_path, TrainingManager.BEST_MODEL_NAME
        )
        model.learn(
            total_timesteps=5_000,
            callback=callback,
            tb_log_name=f"v{training.version}",
            reset_num_timesteps=False,
        )
        model.save(training.parameters.log_dir_path / TrainingManager.LATEST_MODEL_NAME)
        del model

    def stop_training(self) -> TrainingResult:
        return TrainingResult()
