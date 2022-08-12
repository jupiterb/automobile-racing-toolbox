from pydantic import BaseModel, PositiveFloat, PositiveInt


class DQNConfig(BaseModel):
    policy: str 
    total_timesteps: PositiveInt
    buffer_size: PositiveInt 
    learning_starts: PositiveInt
    gamma: PositiveFloat(le=1)
    exploration_final_epsilon: PositiveFloat(le=1)
    learning_rate: PositiveFloat
