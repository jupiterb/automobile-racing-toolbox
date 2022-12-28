import logging
from pydantic import BaseModel
from httpx_oauth.oauth2 import OAuth2Token
import time
from threading import Thread

from src.utils.schemas import Token
from src.utils.exceptions import RegistryAppException
from src.aws.iam_manager import IAMManager


logger = logging.getLogger(__name__)


class TokenInfo(BaseModel):
    token: OAuth2Token
    owner: str


class AccessManager:
    def __init__(self, iam: IAMManager) -> None:
        self._iam = iam
        self._active_tokens: dict[str, TokenInfo] = {}
        self._removing_expired_running = False
        self._removing_expired: Thread

    def give_access(self, username: str, token: Token):
        if not self._removing_expired_running:
            raise RegistryAppException(
                "Cannot give access to user if expired tokens are not removing."
            )
        self._active_tokens[token.token["access_token"]] = TokenInfo(
            token=OAuth2Token(token.token), owner=username
        )
        self._iam.add_user_to_access_group(username)

    def validate_token(self, token: Token) -> str:
        access_token = token.token["access_token"]
        if access_token not in self._active_tokens:
            raise RegistryAppException("Token without access.")
        ouath_token = OAuth2Token(token.token)
        if ouath_token.is_expired():
            raise RegistryAppException("Token is expired.")
        return self._active_tokens[access_token].owner

    def start_give_access(self):
        self._removing_expired_running = True
        self._removing_expired_thread = Thread(target=self._removing_expired_tokens)
        self._removing_expired_thread.start()

    def remove_access_everyone(self):
        self._removing_expired_running = False
        self._removing_expired_thread.join()
        for token in self._active_tokens.values():
            self._iam.remove_user_from_access_group(token.owner)

    def _removing_expired_tokens(self):
        while self._removing_expired_running:
            expired_tokens = self._active_tokens.copy()
            for acceess_token, token in expired_tokens.items():
                if token.token.is_expired():
                    self._iam.remove_user_from_access_group(token.owner)
                else:
                    del expired_tokens[acceess_token]
            expired_owners = []
            for acceess_token, token in expired_tokens.items():
                del self._active_tokens[acceess_token]
                expired_owners.append(token.owner)
            if any(expired_owners):
                logger.info(f"Tokens expired for users: {expired_owners}")
            time.sleep(10)
