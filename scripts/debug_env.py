import logging
from collections import namedtuple
from ray.rllib.env.policy_client import PolicyClient
from racing_toolbox.environment.builder import setup_env
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface.config import GameConfiguration
import wandb 
import time 
import os 


class DebugWorker:
    def __init__(
        self, game_conf: GameConfiguration, env_conf: EnvConfig
    ):
        self.env = setup_env(game_conf, env_conf)

    def run(self) -> None:
        obs = self.env.reset()
        rewards = 0.0
        print(self.env.action_space)
        while True:
            a = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(a)
            rewards += reward

            if done:
                print(f"Total reward: {rewards}")
                rewards = 0.0
                obs = self.env.reset()

if __name__ == "__main__":
    game_conf_path = "C:\\Users\\yogam\\projects\\automobile-racing-toolbox\\config\\trackmania\\game_config.json"
    env_conf_path = "C:\\Users\\yogam\\projects\\automobile-racing-toolbox\\config\\trackmania\\env.json"
    worker = DebugWorker(GameConfiguration.parse_file(game_conf_path), EnvConfig.parse_file(env_conf_path))
    worker.run()