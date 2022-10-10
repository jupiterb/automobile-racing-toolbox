import logging
from collections import namedtuple
from ray.rllib.env.policy_client import PolicyClient
from racing_toolbox.environment.builder import setup_env
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface.config import GameConfiguration

logger = logging.getLogger(__name__)


Address = namedtuple("Address", ["host", "port"])


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
            obs, reward, done, info = self.env.step(action)
            rewards += reward
            self.client.log_returns(eid, reward, info=info)

            if done:
                logger.info("Total reward:", rewards)
                rewards = 0.0
                self.client.end_episode(eid, obs)

                obs = self.env.reset()
                eid = self.client.start_episode(training_enabled=True)
