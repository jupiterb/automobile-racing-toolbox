import streamlit as st
from typing import Optional
import asyncio
from httpx_oauth.clients.google import GoogleOAuth2

from httpx_oauth.oauth2 import OAuth2Token
from ui_app.src.shared import Shared
from ui_app.src.services.registry_service import RegistryServiceException


def log_in() -> Optional[OAuth2Token]:
    st.header("Welcome to Automobile Training Application")

    shared = Shared()

    token = log_in_with_google_oauth()
    if token:
        shared.just_logged = True
        return token

    create_account()

    st.write("Log in")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Log in"):
        try:
            token = shared.registry_service.get_access(username, password)
        except RegistryServiceException as e:
            st.warning(f"{e} \nTry again")
            return
        shared.just_logged = True
        return token


def create_account():
    with st.expander("Don't have account? Create now!"):
        username = st.text_input("Username ")
        email = st.text_input("Email ")
        st.write(
            "For now, username and email are constant and cannot be changed after account creation."
        )

        password = st.text_input("Password ", type="password")
        repeated_password = st.text_input("Repeat password ", type="password")
        st.write(
            "Password should be at least 10 characters long, contain at least one number, one letter and special character."
        )

        wandb_api_key = st.text_input("Weights and Biases API key")

        if password != repeated_password:
            st.warning("Passwords do not match")
            return

        shared = Shared()
        if st.button("Create account"):
            try:
                shared.registry_service.add_user(
                    username, password, email, wandb_api_key
                )
                st.write("Great, now you have your account!")
            except RegistryServiceException as e:
                st.warning(f"Cannot create user: {e.message}")


def log_in_with_google_oauth():
    with st.expander("Continue with google account"):
        client = GoogleOAuth2(
            "1035298897397-kgm7bmv0upcpssmi0g97epsfu6g6hdim.apps.googleusercontent.com",
            "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",
        )
        client.access_token_endpoint
        authorization_url = asyncio.run(
            write_authorization_url(client=client, redirect_uri="http://localhost:8080")
        )
        if (code := st.experimental_get_query_params().get("code")) is None:
            st.markdown(
                f"###### [![this is an image link](https://i.imgur.com/mQAQwvt.png)]({authorization_url})"
            )
        else:
            token = Shared().registry_service.get_google_access(code[0])
            if token.is_expired():
                pass
            else:
                return token


async def write_authorization_url(client: GoogleOAuth2, redirect_uri: str):
    authorization_url = await client.get_authorization_url(
        redirect_uri,
        scope=["profile", "email"],
        extras_params={"access_type": "offline"},
    )
    return authorization_url
