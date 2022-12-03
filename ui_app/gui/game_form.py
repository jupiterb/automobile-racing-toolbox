import streamlit as st
import streamlit_pydantic as sp
from typing import Optional

from racing_toolbox.interface.config import GameConfiguration

from ui_app.config_source.file_based import FileSysteConfigSource


def configure_game() -> Optional[GameConfiguration]:
    source = FileSysteConfigSource[GameConfiguration]("./config/games")
    games = source.get_configs()
    selected = st.selectbox("Select game", list(games.keys()))
    with st.expander("Add new game"):
        name = st.text_input("Provide new game configuration name", value="My_New_Game")
        config = sp.pydantic_form(key="new game config", model=GameConfiguration)
        if config is not None:
            source.add_config(name, config)
    if selected is not None:
        config = games[selected]
        return config
