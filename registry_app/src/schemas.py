from pydantic import BaseModel, validator


class Credentials(BaseModel):
    username: str
    password: str


class UserDefinedData(Credentials):
    email: str
    wandb_api_key: str


class UserData(UserDefinedData):
    user_key_id: str
    user_secret_key: str


class UserModificationRequest(BaseModel):
    credentials: Credentials
    new: UserDefinedData


# TODO validation
