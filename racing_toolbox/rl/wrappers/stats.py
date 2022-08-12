import wandb
import gym


class WandbWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, sync_step: int = 1) -> None:
        super().__init__(env)
        self.sync_step = sync_step
        self._steps_without_sync = 0
        self._log_buffer = []

    def step(self, action):
        observation, reward, done, info = super().step(action)
        if done:
            self._log(info)
        return observation, reward, done, info

    def _log(self, info: dict) -> None:
        """This method is very basic so far, but could be extended to log
        more complicated statistics based on information passed in info dict"""

        self._steps_without_sync += 1
        if self._steps_without_sync == self.sync_step:
            wandb.log(info)
            print(f"logged {info}")
            self._steps_without_sync = 0
