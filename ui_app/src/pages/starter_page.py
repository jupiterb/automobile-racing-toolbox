import streamlit as st

from ui_app.src.forms import (
    configure_game,
    configure_env,
    configure_training,
    configure_pretained,
    review_all,
    start_training,
)
from ui_app.src.forms.common import configure_screen_frame, select_recordings
from ui_app.src.config import (
    UIAppConfig,
    ConfigSourceType,
    Credentials,
    ConfigsContentEndpoints,
)
from ui_app.src.shared import Shared


def select_training_way() -> str:
    st.header("Select way to train")
    option = st.selectbox(
        "Would you like to use pretrained model or create new one from scratch?",
        ["From scratch", "Use pretrained", "Train autoencoder"],
    )
    return option


def configure_from_scratch():
    st.markdown("""---""")
    st.header("Configure training")
    game_config = configure_game()
    env_config = configure_env(game_config)
    training_config = configure_training()
    if game_config and env_config and training_config:
        game_config, env_config, training_config = review_all(
            game_config, env_config, training_config
        )
        start_training(game_config, env_config, training_config)


def configure_with_pretrained():
    st.markdown("""---""")
    st.header("Let's train again")
    pretrained = configure_pretained()
    if pretrained:
        game_config, env_config, training_config = pretrained
        game_config, env_config, training_config = review_all(
            game_config, env_config, training_config
        )
        start_training(game_config, env_config, training_config)


def configure_autencoder():
    st.markdown("""---""")
    st.header("Seems you first want to create special feature extractor. Great choice!")
    game_config = configure_game()
    st.markdown("""---""")
    st.write("Configure autoencoder training")
    screen_frame = configure_screen_frame()
    recordings_to_use = select_recordings()


def setup_shared():
    shared = Shared()
    shared.use_config(
        UIAppConfig(
            credentials=Credentials(
                user_key_id="AKIAVEKIFJHIO7KY23YZ",
                user_secret_key="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",
            ),
            source_type=ConfigSourceType.S3_BUCKET,
            root="automobile-training-test",
            endpoints=ConfigsContentEndpoints(
                game_configs="configs/games",
                env_configs="configs/envs",
                training_configs="configs/trainings",
                recordings="recordings",
            ),
        )
    )


def main():
    st.markdown(
        f'<h1 style="color:#ee6c4d;font-size:40px;">{"Automobile Training Starter"}</h1>',
        unsafe_allow_html=True,
    )

    setup_shared()

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
