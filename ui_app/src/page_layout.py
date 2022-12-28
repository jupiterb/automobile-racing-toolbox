import streamlit as st
from streamlit.runtime.legacy_caching import clear_cache
from typing import Optional, Callable
from httpx_oauth.oauth2 import OAuth2Token
from ui_app.src.config import UserData

from ui_app.src.shared import Shared
from ui_app.src.forms import log_in


def racing_toolbox_page_layout(title: str, content: Callable):
    stylish(title)

    shared = Shared()
    shared.just_logged = False
    shared.is_google_user = False

    token = authorize()

    if token and not token.is_expired():
        if shared.just_logged:
            shared.just_logged = False
            st.experimental_rerun()
        shared.token = token
        shared.user_data = get_user_data(token)
        shared.is_google_user = not any(shared.user_data.password)
        content()
    else:
        clear_cache()


def stylish(title: str):
    st.set_page_config(page_title=title, layout="wide")
    custom_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer:after{content: ' by jupiterb and czyjtu'; color:#ee6c4d; font-size:15px;}
            </style>
            """
    st.markdown(
        f'<h1 style="color:#ee6c4d;font-size:40px;">{title}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("""---""")
    st.markdown(custom_style, unsafe_allow_html=True)


@st.cache(suppress_st_warning=True)
def authorize() -> Optional[OAuth2Token]:
    shared = Shared()
    try:
        return shared.token
    except:
        token = log_in()
        if token:
            return token


@st.cache(suppress_st_warning=True)
def get_user_data(token: OAuth2Token) -> UserData:
    return Shared().registry_service.get_data(token)
