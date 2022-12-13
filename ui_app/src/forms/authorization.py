import streamlit as st
from typing import Optional

from ui_app.src.config import UserData
from ui_app.src.shared import Shared


def log_in() -> Optional[UserData]:
    st.header("Welcome to Automobile Training Application")
    sign_in()
    st.write("Log in")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    shared = Shared()
    if st.button("Log in"):
        user_data = shared.registry_service.get_access(username, password)
        if user_data:
            shared.just_logged = True
            return user_data
        else:
            st.warning("Wrong credentials. Try again")


def sign_in():
    with st.expander("Don't have account? Sign in now!"):
        username = st.text_input("Username ")
        email = st.text_input("Email ")
        password = st.text_input("Password ", type="password")
        wandb_api_key = st.text_input("Weights and Biases API key")

        shared = Shared()
        if st.button("Sign in"):
            if shared.registry_service.add_user(
                username, password, email, wandb_api_key
            ):
                st.write("Great, now you have your account!")
            else:
                st.warning("User with same username already exists")
