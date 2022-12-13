import streamlit as st

from ui_app.src.forms import (
    configure_training_resuming,
    configure_training,
    review_config,
    resume_training,
)
from ui_app.src.shared import Shared
from ui_app.src.page_layout import racing_toolbox_page_layout


def main():
    st.header(
        "Here you can start new training using one of yours checkpoints from Weight and Biases"
    )
    st.write("You can find your checkpoints there")
    link = "[Weights&Biases](https://wandb.ai/site)"
    st.markdown(link, unsafe_allow_html=True)

    st.write(
        "Before you run training, make sure you have valid API KEY in account page."
    )

    st.markdown("""---""")

    shared = Shared()

    rub_reference, checkpoint_name = configure_training_resuming()
    # TODO get user api key
    # TODO get training config from checkpoint
    original_config = list(shared.training_configs_source.get_configs().items())[0][1]

    st.markdown("""---""")
    st.write(
        "You can use one of your training configurations or create one. Original model configuration won't be overrited."
    )
    selected_config = configure_training()
    if selected_config is not None:
        original_model = original_config.model
        selected_config.model = original_model
        original_config = selected_config

    with st.sidebar.header("Review configuration"):
        original_config = review_config(
            original_config, "Training configuration", shared.training_configs_source
        )

    resume_training(original_config, "USER KEY", rub_reference, checkpoint_name)


if __name__ == "__main__":
    racing_toolbox_page_layout("Automobile Training Checkpoints", main)
