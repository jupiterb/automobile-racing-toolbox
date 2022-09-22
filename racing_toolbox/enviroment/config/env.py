from pydantic import BaseModel, PositiveInt

from racing_toolbox.enviroment.config.reward import RewardConfig
from racing_toolbox.enviroment.config.observation import ObservationConfig


class EnvConfig(BaseModel):
    reward_config: RewardConfig
    observation_config: ObservationConfig
    max_episode_length: PositiveInt
