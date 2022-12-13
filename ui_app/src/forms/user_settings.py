import streamlit as st
from typing import Optional

from ui_app.src.config import UserData
from ui_app.src.shared import Shared


def review_account_settings() -> bool:
    """retuens True if user has account / is logged, else False"""
    shared = Shared()

    user_data = shared.user_data

    user_name = user_data.username
    password = user_data.password

    st.header(f"Hello {user_name}")
    st.write("Here you can manage your account")

    change_wandb_api_key()
    change_password()

    return True


def change_password():
    shared = Shared()
    current_user_data = shared.user_data
    new_password = ask_for_new_credential("password", current_user_data.password, True)
    if new_password:
        new_user_data = current_user_data.copy()
        new_user_data.password = new_password
        modify_user_data(current_user_data, new_user_data)


def change_wandb_api_key():
    shared = Shared()
    current_user_data = shared.user_data
    with st.expander("Check your Weight and Biases API key"):
        st.write(current_user_data.wandb_api_key)
    new_wandb_api_key = ask_for_new_credential(
        "Weight and Biases API key", current_user_data.password, False
    )
    if new_wandb_api_key:
        new_user_data = current_user_data.copy()
        new_user_data.wandb_api_key = new_wandb_api_key
        modify_user_data(current_user_data, new_user_data)


def modify_user_data(current: UserData, new: UserData):
    result = Shared().registry_service.modify_user(current, new)
    if result is None:
        return
    st.warning(f"Cannot modify user data: {result}")


def ask_for_new_credential(
    credential_name: str, current_password: str, hide_typing: bool
) -> Optional[str]:
    """returns new value if user provide valid password"""
    _type = "default" if not hide_typing else "password"
    with st.expander(f"Change your {credential_name}"):
        new_value = st.text_input(f"Provide new {credential_name}", type=_type)
        st.markdown("""---""")
        password = st.text_input(
            f"Provide your current password to change {credential_name}",
            type="password",
        )
        if st.button(f"Change {credential_name}"):
            if not len(password):
                st.warning("Empty value")
                return
            if password != current_password:
                st.warning("Wrong password!")
                return
            return new_value
