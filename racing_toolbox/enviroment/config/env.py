from pydantic import BaseModel, PositiveInt

from enviroment.config.reward import RewardConfig
from enviroment.config.observation import ObservationConfig


class EnvConfig(BaseModel):
    reward_config: RewardConfig
    observation_config: ObservationConfig
    max_episode_length: PositiveInt
