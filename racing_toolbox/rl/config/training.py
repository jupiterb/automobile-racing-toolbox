from pydantic import BaseModel, PositiveFloat, PositiveInt


class BaseTrainingConfig(BaseModel):
    policy: str
    total_timesteps: PositiveInt
    buffer_size: PositiveInt
    learning_starts: PositiveInt
    gamma: PositiveFloat(le=1)
    learning_rate: PositiveFloat


class DQNConfig(BaseTrainingConfig):
    exploration_final_epsilon: PositiveFloat(le=1)


class SACConfig(BaseTrainingConfig):
    pass
