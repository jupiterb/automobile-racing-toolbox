import streamlit as st
from streamlit.runtime.legacy_caching import clear_cache
from typing import Optional, Callable

from ui_app.src.shared import Shared
from ui_app.src.config import UserData
from ui_app.src.forms import log_in


def racing_toolbox_page_layout(title: str, content: Callable):
    stylish(title)

    shared = Shared()
    shared.just_logged = False

    auth_result = authorize()

    if auth_result:
        if shared.just_logged:
            shared.just_logged = False
            st.experimental_rerun()
        Shared().user_data = auth_result
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
def authorize() -> Optional[UserData]:
    shared = Shared()
    try:
        return shared.user_data
    except:
        data = log_in()
        if data:
            return data
