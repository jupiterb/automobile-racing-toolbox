import argparse
import wandb
from racing_toolbox.training.config.params import TrainingParams
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.environment.mocked import MockedEnv
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.training.config.user_defined import TrainingConfig
from racing_toolbox.environment import builder
from racing_toolbox.environment.builder import setup_env
from racing_toolbox.training import Trainer
from pathlib import Path


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        "-r",
        type=str,
        help="W&B run path",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        help="W&B checkpoint name (with tag)",
    )
    return parser.parse_args()


def get_training_params(
    game_config: GameConfiguration,
    env_config: EnvConfig,
    training_config: TrainingConfig,
):
    actions = (
        game_config.discrete_actions_mapping
        if env_config.action_config.available_actions
        else game_config.continous_actions_mapping
    )
    mocked_env = MockedEnv(actions, game_config.window_size)
    env = builder.wrapp_env(mocked_env, env_config)

    trainer_params = TrainingParams(
        **training_config.dict(),
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    return trainer_params


class WandbWorker:
    def __init__(self, run_ref: str, checkpoint_name: str):
        # create run, download files, and models, setup env, build model
        with wandb.init(project="ART") as run:
            checkpoint_ref = f"{'/'.join(run_ref.split('/')[:-1])}/{checkpoint_name}"
            checkpoint_artefact = run.use_artifact(checkpoint_ref, type="checkpoint")
            checkpoint_dir = checkpoint_artefact.download()
            game_config = GameConfiguration.parse_file(
                wandb.restore("game_config.json", run_path=run_ref).name
            )
            env_config = EnvConfig.parse_file(
                wandb.restore("env_config.json", run_path=run_ref).name
            )
            training_config = TrainingConfig.parse_file(
                wandb.restore("training_config.json", run_path=run_ref).name
            )
        trainer_params = get_training_params(game_config, env_config, training_config)
        self.algorithm = Trainer(
            trainer_params,
            Path(checkpoint_dir).absolute() / "checkpoint",
        ).algorithm
        env_config.max_episode_length = int(1e10)
        self.env = setup_env(game_config, env_config)

    def run(self) -> None:
        obs = self.env.reset()
        rewards = 0.0
        while True:
            a = self.algorithm.compute_single_action(obs, explore=False)
            obs, reward, done, info = self.env.step(a)
            rewards += reward

            if done:
                print(f"Total reward: {rewards}")
                rewards = 0.0
                obs = self.env.reset()


def main():
    args = get_cli_args()
    worker = WandbWorker(args.run, args.checkpoint)
    worker.run()
