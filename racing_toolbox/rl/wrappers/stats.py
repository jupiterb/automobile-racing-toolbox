import wandb 
import gym 


class WandbWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, sync_step: int=1) -> None:
        super().__init__(env)
        self.sync_step = sync_step
        self._steps_without_sync = 0 
        self._log_buffer = []
    
    def step(self, action):
        o, r, done, info = super().step(action)
        if done:
            self._log(info)
        return o, r, done, info 

    def _log(self, info: dict) -> None:
        self._steps_without_sync += 1
        if self._steps_without_sync == self.sync_step:
            wandb.log(info)
            print(f"logged {info}")
            self._steps_without_sync = 0
