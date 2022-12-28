from pydantic import BaseModel, root_validator, validator
from typing import Any, Optional
import string


class Credentials(BaseModel):
    username: str
    password: str = ""


class UserDefinedData(Credentials):
    email: str
    wandb_api_key: str = ""


class UserFullData(UserDefinedData):
    user_key_id: str
    user_secret_key: str


class GetTokenRequest(BaseModel):
    credentials: Optional[Credentials]
    code: Optional[str]

    @root_validator(pre=True)
    def check_card_number_omitted(cls, values):
        credentials, code = values.get("credentials"), values.get("code")
        assert credentials or code, "Credentials or code should be provided"
        return values


class Token(BaseModel):
    token: dict[str, Any]


class UserModificationRequest(BaseModel):
    token: Token
    new: UserDefinedData
