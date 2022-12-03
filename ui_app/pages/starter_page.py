import streamlit as st

from ui_app.gui.game_form import configure_game
from ui_app.gui.env_form import configure_env, screen_frame_form
from ui_app.gui.training_form import configure_training, recordings_form
from ui_app.gui.pretrained_form import configure_pretained
from ui_app.gui.review_config import review_configs
from ui_app.gui.starter_form import starter_form


def select_training_way() -> str:
    st.header("Select way to train new amazing AI")
    option = st.selectbox(
        "Would you like to use pretrained model or create new one from scratch?",
        ["From scratch", "Use pretrained", "Train autencoder"],
    )
    return option


def configure_from_scratch():
    st.markdown("""---""")
    st.header("Configure training")
    game_config = configure_game()
    env_config = configure_env(game_config)
    training_config = configure_training()
    if game_config and env_config and training_config:
        game_config, env_config, training_config = review_configs(
            game_config, env_config, training_config
        )
        starter_form(game_config, env_config, training_config)


def configure_with_pretrained():
    st.markdown("""---""")
    st.header("Let's train again")
    pretrained = configure_pretained()
    if pretrained:
        game_config, env_config, training_config = pretrained
        game_config, env_config, training_config = review_configs(
            game_config, env_config, training_config
        )
        starter_form(game_config, env_config, training_config)


def configure_autencoder():
    st.markdown("""---""")
    st.header("Seems you first want to create special feature extractor. Great choice!")
    game_config = configure_game()
    st.markdown("""---""")
    st.write("Configure autoencoder training")
    screen_frame = screen_frame_form()
    recordings_to_use = recordings_form()


def main():
    st.markdown(
        f'<h1 style="color:#ff8833;font-size:40px;">{"Automobile Training Starter"}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("""---""")

    training_type = select_training_way()
    if training_type == "From scratch":
        configure_from_scratch()
    elif training_type == "Use pretrained":
        configure_with_pretrained()
    else:
        configure_autencoder()


if __name__ == "__main__":
    st.set_page_config(page_title="Automobile Training Starter", layout="wide")
    main()
