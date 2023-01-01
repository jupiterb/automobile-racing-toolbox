import logging
from collections import namedtuple
from ray.rllib.env.policy_client import PolicyClient
from racing_toolbox.environment.builder import setup_env
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.interface.controllers.gamepad import GamepadController
import vgamepad as vg
import wandb
import time
import os

logger = logging.getLogger(__name__)


Address = namedtuple("Address", ["host", "port"])


GamepadController.global_gamepad = vg.VX360Gamepad()


def run_worker_process(
    policy_address, game_config, env_config, wandb_api_key, wandb_project, wandb_group
):
    os.environ["WANDB_API_KEY"] = wandb_api_key
    with wandb.init(project=wandb_project, group=wandb_group) as run:
        worker = Worker(
            policy_address=policy_address,
            game_conf=game_config,
            env_conf=env_config,
        )
        time.sleep(5)
        worker.run()


class Worker:
    def __init__(
        self, policy_address: Address, game_conf: GameConfiguration, env_conf: EnvConfig
    ):
        self.env = setup_env(game_conf, env_conf)
        self.client = PolicyClient(
            address=f"http://{policy_address.host}:{policy_address.port}",
            inference_mode="remote",
        )

    def run(self) -> None:
        logger.info("starting rollouts")

        obs = self.env.reset()
        eid = self.client.start_episode(training_enabled=True)
        rewards = 0.0
        while True:
            action = self.client.get_action(eid, obs)
            print("applying action")
            obs, reward, done, info = self.env.step(action)
            rewards += reward
            print("loggin returns")
            self.client.log_returns(eid, reward, info=info)

            if done:
                logger.info(f"Total reward: {rewards}")
                rewards = 0.0
                self.client.end_episode(eid, obs)

                obs = self.env.reset()
                eid = self.client.start_episode(training_enabled=True)
