import streamlit as st
import wandb
import os

from racing_toolbox.training.config import TrainingConfig

from ui_app.src.forms import (
    configure_training_resuming,
    configure_training,
    review_config,
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

    run_reference, checkpoint_name = configure_training_resuming()
    wandb_api_key = shared.user_data.wandb_api_key
    original_config = _get_training_config_from_checkpoint(
        run_reference, checkpoint_name
    )

    if original_config is None:
        st.warning("There is no training configuration under this checkpoint.")
        return

    st.markdown("""---""")
    st.write(
        "You can use one of your training configurations or create one. Original model configuration won't be overrited."
    )
    training_config = configure_training()
    if training_config is None:
        return

    original_model = original_config.model
    training_config.model = original_model

    with st.sidebar.header("Review configuration"):
        training_config = review_config(
            training_config, "Training configuration", shared.training_configs_source
        )

    resume_training(training_config, wandb_api_key, run_reference, checkpoint_name)


def _get_training_config_from_checkpoint(run_reference: str, checkpoint_name: str):
    try:
        with wandb.init(project="ART") as run:
            checkpoint_ref = (
                f"{'/'.join(run_reference.split('/')[:-1])}/{checkpoint_name}"
            )
            print(checkpoint_ref)
            checkpoint_artefact = run.use_artifact(checkpoint_ref, type="checkpoint")
            checkpoint_dir = checkpoint_artefact.download()
            training_config = TrainingConfig.parse_file(
                wandb.restore("training_config.json", run_path=run_reference).name
            )
            return training_config
    except:
        return None


def configure_training_resuming() -> tuple[str, str]:
    """Returns wandb run reference and checkpoint name"""
    st.write("Provide Weights and Biases run reference")
    run_reference = st.text_input("Run reference")

    st.write("Provide Weights and Biases checkpoint name")
    checkpoint_name = st.text_input("Checkpoint name")

    return run_reference, checkpoint_name


def resume_training(
    training_config: TrainingConfig,
    wandb_key: str,
    wandb_run_reference: str,
    checkpoint_name: str,
):
    st.markdown("""---""")
    st.header("Confirm start of training")
    if st.button("Run"):
        Shared().trainer_service.resume_training(
            training_config, wandb_key, wandb_run_reference, checkpoint_name
        )


if __name__ == "__main__":
    racing_toolbox_page_layout("Automobile Training Checkpoints", main)
