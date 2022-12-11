from pydantic import BaseModel
from typing import Optional
from ray.rllib.algorithms import Algorithm
import wandb, os
from pathlib import Path
from ray.rllib.env.policy_server_input import PolicyServerInput
from racing_toolbox.training.config.params import TrainingParams
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.environment.mocked import MockedEnv
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.training.config.user_defined import TrainingConfig
from racing_toolbox.environment import builder


class WorkerFailure(Exception):
    def __init__(self, worker_address: str, reason: str, details: Optional[str] = None):
        super().__init__()
        self.worker_address = worker_address
        self.reason = reason
        self.details = details


def wandb_checkpoint_callback_factory(checkpoint_artifact: wandb.Artifact, dir: Path):
    def callback(algorithm: Algorithm):
        chkpnt_path = algorithm.save(str(dir))
        checkpoint_artifact.add_dir(chkpnt_path, name="checkpoint")
        wandb.log_artifact(checkpoint_artifact)

    return callback


def log_config(config: BaseModel, name: str) -> None:
    filename = Path(name + ".json")
    with open(filename, "w") as f:
        f.write(config.json())
    wandb.save(str(filename), policy="now")
    # filename.unlink()


def get_training_params(
    host: str,
    port: int,
    training_config: TrainingConfig,
    game_config: GameConfiguration,
    env_config: EnvConfig,
):
    def input_(ioctx):
        if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
            return PolicyServerInput(
                ioctx,
                host,
                port + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
            )
        else:
            return None

    # TODO: How to choose correct interface action mapping based only on game config?
    mocked_env = MockedEnv(
        game_config.discrete_actions_mapping, game_config.window_size
    )
    env = builder.wrapp_env(mocked_env, env_config)
    print(f"observation space: {env.observation_space.shape}")
    trainer_params = TrainingParams(
        **training_config.dict(),
        observation_space=env.observation_space,
        action_space=env.action_space,
        input_=input_,
    )
    return trainer_params
