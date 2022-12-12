from pydantic import BaseModel 


class VAEConfig(BaseModel):
    wandb_checkpoint_ref: str 
