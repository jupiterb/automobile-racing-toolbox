from pickle import dumps
from ray.rllib.env.policy_server_input import PolicyServerInput
from racing_toolbox.training.config.params import TrainingParams
from racing_toolbox.environment import builder
from racing_toolbox.training import Trainer, config, trainer

from racing_toolbox.training.config.user_defined import TrainingConfig
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface.config import GameConfiguration

import json
from multiprocessing import Process


def _training_thread(at):
    def input_(ioctx):
        if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
            return PolicyServerInput(
                ioctx,
                host,
                port + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
            )
        else:
            return None

    trainer_params = TrainingParams(
        **training_config.dict(),
        env=builder.setup_env(game_config, env_config),
        input_=input_,
    )
    trainer = Trainer(trainer_params)
    trainer.run()


if __name__ == "__main__":

    train_path = (
        "C:\\Users\\yogam\\projects\\automobile-racing-toolbox\\training_config.json"
    )
    game_path = (
        "C:\\Users\\yogam\\projects\\automobile-racing-toolbox\\game_config.json"
    )
    env_path = "C:\\Users\\yogam\\projects\\automobile-racing-toolbox\\env.json"

    with open(game_path, "r") as gp, open(env_path, "r") as ep, open(
        train_path, "r"
    ) as tp:
        game_config = GameConfiguration(**json.load(gp))
        env_config = EnvConfig(**json.load(ep))
        training_config = TrainingConfig(**json.load(tp))

    trainer_process = Process(
        target=_training_thread,
        args=(
            dumps(
                (
                    game_config,
                    env_config,
                    training_config,
                    "0.0.0.0",
                    8000,
                )
            ),
        ),
    )
    trainer_process.start()
    trainer_process.join()
