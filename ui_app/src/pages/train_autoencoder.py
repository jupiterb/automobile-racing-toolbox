import streamlit as st

from racing_toolbox.observation.config.vae_config import (
    VAETrainingConfig,
    VAEModelConfig,
)

from ui_app.src.forms import (
    review_config,
    configure_encoder,
    configure_vae_training,
)
from ui_app.src.forms.common import select_recordings
from ui_app.src.shared import Shared
from ui_app.src.config import AppConfig
from ui_app.src.page_layout import racing_toolbox_page_layout


def main():
    st.header("Configure training")
    shared = Shared()

    recordings = select_recordings()
    model = configure_encoder()
    training_config = configure_vae_training()

    if any(recordings) and model and training_config:
        with st.sidebar.header("Review configuration"):
            model = review_config(model, "VAE model", shared.games_source)
            training_config = review_config(
                training_config,
                "VAE training configuration",
                shared.vae_training_configs_source,
            )
        wandb_api_key = shared.user_data.wandb_api_key
        train_autoencoder(
            wandb_api_key,
            training_config,
            model,
            AppConfig().bucket_name,
            recordings,
        )


def train_autoencoder(
    wandb_key: str,
    training_params: VAETrainingConfig,
    encoder_config: VAEModelConfig,
    bucket_name: str,
    recordings_refs: list[str],
):
    st.markdown("""---""")
    st.header("Confirm start of training")
    if st.button("Run"):
        Shared().trainer_service.start_autoencoder_training(
            wandb_key, training_params, encoder_config, bucket_name, recordings_refs
        )


if __name__ == "__main__":
    racing_toolbox_page_layout("Automobile Autoencoder Training Starter", main)
