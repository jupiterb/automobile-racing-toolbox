import streamlit as st

from ui_app.src.forms import (
    configure_game,
    review_config,
    configure_vae,
    train_autoencoder,
)
from ui_app.src.forms.common import select_recordings
from ui_app.src.shared import Shared
from ui_app.src.page_layout import racing_toolbox_page_layout


def main():
    st.header("Configure training")

    game_config = configure_game()
    recordings = select_recordings()
    configure_vae()

    if game_config:
        st.sidebar.markdown("""---""")
        with st.sidebar.header("Review configuration"):
            review_config(game_config, "Game configuration", Shared().games_source)

    train_autoencoder()


if __name__ == "__main__":
    racing_toolbox_page_layout("Automobile Autoencoder Training Starter", main)
