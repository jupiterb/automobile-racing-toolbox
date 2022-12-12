from pydantic import BaseModel 


class VAEConfig(BaseModel):
    wandb_api_key: str # TODO: consider removing this from config, and forcing user to use only one account 
    wandb_checkpoint_ref: str 
