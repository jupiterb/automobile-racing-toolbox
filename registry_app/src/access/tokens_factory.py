import asyncio
from httpx_oauth.clients.google import GoogleOAuth2
from httpx_oauth.oauth2 import OAuth2Token
import uuid

from src.utils.schemas import Token, Credentials, UserDefinedData


class TokensFactory:
    def __init__(
        self, redirect_uri: str, google_client_id: str, google_client_secret: str
    ) -> None:
        self._google_client = GoogleOAuth2(google_client_id, google_client_secret)
        self._redirect_uri = redirect_uri

    def generate_token(self, credentials: Credentials) -> Token:
        access_token = str(uuid.uuid4())
        oauth_token = OAuth2Token(
            {
                "access_token": access_token,
                "expires_in": 3599,
                "owner": credentials.username,
                "secret": credentials.password,
            }
        )
        return Token(token=oauth_token)

    def generate_google_token(self, code: str) -> Token:
        token = asyncio.run(self._write_google_access_token(code=code))
        return Token(token=token)

    def generate_google_user_data(self, token: Token) -> UserDefinedData:
        user_id, user_email = asyncio.run(
            self._get_google_email(token=token.token["access_token"])
        )
        username = user_id.split("/")[1]
        return UserDefinedData(username=username, email=user_email)

    async def _write_google_access_token(self, code):
        token = await self._google_client.get_access_token(code, self._redirect_uri)
        return token

    async def _get_google_email(self, token):
        user_id, user_email = await self._google_client.get_id_email(token)
        return user_id, user_email
