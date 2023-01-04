import streamlit as st
from streamlit.runtime.legacy_caching import clear_cache
from typing import Optional

from ui_app.src.page_layout import racing_toolbox_page_layout
from ui_app.src.config import UserData
from ui_app.src.services.registry_service import RegistryServiceException
from ui_app.src.shared import Shared


def main():
    shared = Shared()
    user_data = shared.user_data
    username = (
        user_data.username
        if not shared.is_google_user
        else user_data.email.split("@")[0]
    )

    st.header(f"Hello {username}")
    st.write("Here you can manage your account")

    change_wandb_api_key()
    if not shared.is_google_user:
        change_password()


def change_password():
    shared = Shared()
    user_data = shared.user_data
    new_data = ask_for_new_data("password", True)
    if new_data:
        new_user_data = user_data.copy()
        new_user_data.password = new_data
        modify_user_data(new_user_data)


def change_wandb_api_key():
    shared = Shared()
    user_data = shared.user_data
    with st.expander("Check your Weight and Biases API key"):
        st.write(user_data.wandb_api_key)
    new_data = ask_for_new_data(
        "Weight and Biases API key",
    )
    if new_data:
        new_user_data = user_data.copy()
        new_user_data.wandb_api_key = new_data
        modify_user_data(new_user_data)


def modify_user_data(new: UserData):
    shared = Shared()
    try:
        shared.registry_service.modify_user(shared.token, new)
        clear_cache()
        shared.user_data = new
        st.experimental_rerun()
    except RegistryServiceException as e:
        st.warning(f"Cannot modify user data: {e.message}")


def ask_for_new_data(
    data_nama: str,
    hide_typing: bool = False,
) -> Optional[str]:
    _type = "default" if not hide_typing else "password"
    shared = Shared()
    with st.expander(f"Change your {data_nama}"):
        new_value = st.text_input(f"Provide new {data_nama}", type=_type)
        repeated_new_value = (
            st.text_input(f"Confirm new {data_nama}", type=_type)
            if hide_typing
            else new_value
        )
        st.markdown("""---""")
        password_provided = (
            st.text_input(
                f"Provide your current password to change {data_nama}",
                type="password",
            )
            if not shared.is_google_user
            else shared.user_data.password
        )
        if new_value != repeated_new_value:
            st.warning(f"Provided values do not match")
            return
        if st.button(f"Change {data_nama}"):
            if password_provided != shared.user_data.password:
                st.warning("Wrong password.")
                return
            return new_value


if __name__ == "__main__":
    racing_toolbox_page_layout("Your account", main)
