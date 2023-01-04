import json
import requests
from httpx_oauth.oauth2 import OAuth2Token

from ui_app.src.config import UserData


class RegistryServiceException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    @property
    def message(self):
        return self._message


class RegistryService:
    def __init__(self, url: str):
        self._url = url

    def add_user(self, username: str, password: str, email: str, wandb_api_key: str):
        data = {
            "username": username,
            "password": password,
            "email": email,
            "wandb_api_key": wandb_api_key,
        }
        encoded_data = json.dumps(data).encode("utf-8")
        response = requests.post(f"{self._url}/registry/user", data=encoded_data)
        if response.status_code != 200:
            raise RegistryServiceException(response.text)

    def get_access(self, username: str, password: str) -> OAuth2Token:
        data = {"credentials": {"username": username, "password": password}}
        encoded_data = json.dumps(data).encode("utf-8")
        response = requests.get(f"{self._url}/registry/access", data=encoded_data)
        if response.status_code != 200:
            raise RegistryServiceException(response.text)
        return OAuth2Token(dict(**response.json())["token"])

    def get_google_access(self, code: str) -> OAuth2Token:
        data = {"code": code}
        encoded_data = json.dumps(data).encode("utf-8")
        response = requests.get(f"{self._url}/registry/access", data=encoded_data)
        if response.status_code != 200:
            raise RegistryServiceException(response.text)
        return OAuth2Token(dict(**response.json())["token"])

    def get_data(self, token: OAuth2Token) -> UserData:
        data = {"token": token}
        encoded_data = json.dumps(data).encode("utf-8")
        response = requests.get(f"{self._url}/registry/data", data=encoded_data)
        if response.status_code != 200:
            raise RegistryServiceException(response.text)
        return UserData(**response.json())

    def modify_user(self, token: OAuth2Token, new: UserData):
        data = {
            "token": {"token": token},
            "new": {
                "username": new.username,
                "password": new.password,
                "email": new.email,
                "wandb_api_key": new.wandb_api_key,
            },
        }
        encoded_data = json.dumps(data).encode("utf-8")
        response = requests.post(f"{self._url}/registry/modify", data=encoded_data)
        if response.status_code != 200:
            raise RegistryServiceException(response.text)
