import streamlit as st
<<<<<<< HEAD
import asyncio
=======
import os
import asyncio
import traceback
>>>>>>> 2a43dfdba2e67dea4ff8ec1fd72a61921726cc4f
from httpx_oauth.clients.google import GoogleOAuth2
from typing import Optional
from pydantic import BaseSettings
from dataclasses import dataclass


class Settings(BaseSettings):
    client_id: str
    client_secret_key: str
    base_url: str


@dataclass
class UserCredentials:
    client_id: str
    client_email: str
    # token: OAuth2Token


def auth_pannel() -> Optional[UserCredentials]:
    def sign_in_box(authorization_url, key):
        placeholder = st.empty()
        with placeholder.form(
            "Sign In" + key,
        ):
            st.markdown("#### Enter your credentials")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            st.write("or")
            st.markdown(
                f"###### [![this is an image link](https://i.imgur.com/mQAQwvt.png)]({authorization_url})"
            )

    settings = Settings()
    client = GoogleOAuth2(settings.client_id, settings.client_secret_key)
    authorization_url = asyncio.run(
        write_authorization_url(client=client, redirect_uri=settings.base_url)
    )

    if "token" not in st.session_state:
        st.session_state["token"] = None

    if st.session_state.get("token") is None:
        if (code := st.experimental_get_query_params().get("code")) is None:
            sign_in_box(authorization_url, "1")
        else:
            try:
                token = asyncio.run(
                    write_access_token(
                        client=client, redirect_uri=settings.base_url, code=code
                    )
                )
            except Exception as e:
                sign_in_box(authorization_url, "2")
                st.error(
                    "page was reload, or account is not allowed to acces this site"
                )
            else:
                # Check if token has expired:
                if token.is_expired():
                    sign_in_box(authorization_url, "3")
                    st.error("Login session has ended")
                else:
                    st.session_state.token = token
                    user_id, user_email = asyncio.run(
                        get_email(client=client, token=token["access_token"])
                    )
                    user = UserCredentials(user_id, user_email)
                    return user


async def write_authorization_url(client: GoogleOAuth2, redirect_uri: str):
    authorization_url = await client.get_authorization_url(
        redirect_uri,
        scope=["profile", "email"],
        extras_params={"access_type": "offline"},
    )
    return authorization_url


async def write_access_token(client: GoogleOAuth2, redirect_uri, code):
    token = await client.get_access_token(code, redirect_uri)
    return token


async def get_email(client: GoogleOAuth2, token):
    user_id, user_email = await client.get_id_email(token)
    return user_id, user_email
