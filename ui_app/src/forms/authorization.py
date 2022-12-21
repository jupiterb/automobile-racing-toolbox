import streamlit as st
from typing import Optional

from ui_app.src.config import UserData
from ui_app.src.shared import Shared
from ui_app.src.services.registry_service import RegistryServiceException


def log_in() -> Optional[UserData]:
    st.header("Welcome to Automobile Training Application")

    create_account()

    st.write("Log in")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    shared = Shared()
    if st.button("Log in"):
        try:
            user_data = shared.registry_service.get_access(username, password)
        except RegistryServiceException as e:
            st.warning(f"{e} \nTry again")
            return
        shared.just_logged = True
        return user_data


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
