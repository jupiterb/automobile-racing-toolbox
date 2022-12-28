import streamlit as st
from typing import Optional
import json

from racing_toolbox.observation.config.vae_config import (
    VAETrainingConfig,
    VAEModelConfig,
    ConvFilter,
)

from ui_app.src.shared import Shared
from ui_app.src.utils import SetEncoder
from ui_app.src.forms.common import configure_screen_frame


def configure_encoder() -> Optional[VAEModelConfig]:
    source = Shared().vae_models_source
    encoders = source.get_configs()
    selected = st.selectbox(
        "Select variational autencoder model", list(encoders.keys())
    )
    with st.expander("Add new variational autencoder model"):
        name = st.text_input(
            "Provide new variational autoencoder model name",
            value="My_New_Vae_Model",
        )
        with st.form("new_vae_model"):
            model = configure_vae_model()
            submitted = st.form_submit_button("Submit")
            if submitted:
                source.add_config(name, model)
    if selected is not None:
        model = encoders[selected]
        return model


def configure_vae_training():
    source = Shared().vae_training_configs_source
    configs = source.get_configs()
    selected = st.selectbox(
        "Select variational autencoder training config", list(configs.keys())
    )
    with st.expander("Add new variational autencoder training config"):
        name = st.text_input(
            "Provide new variational autoencoder training config name",
            value="My_New_Vae_Training_Config",
        )
        with st.form("new_vae_config"):
            config = configure_vae_training_parameters()
            submitted = st.form_submit_button("Submit")
            if submitted:
                source.add_config(name, config)
    if selected is not None:
        config = configs[selected]
        return config


def configure_vae_model():
    model = VAEModelConfig(
        conv_filters=[ConvFilter(out_channels=4, kernel=(7, 7), stride=1)]
    )
    model_str = st.text_area(
        "Configure model architecture",
        json.dumps(model.dict(), cls=SetEncoder, indent=4),
        height=200,
    )
    try:
        model = VAEModelConfig(**json.loads(model_str))
    except:
        st.warning("Oops. Your model is not valid")
    return model


def configure_vae_training_parameters():
    st.write("Configure training params")
    lr = st.number_input(
        "Learning rate", min_value=1e-5, max_value=1.0, value=1e-4, step=5e-5
    )
    epochs = int(st.number_input("Epochs", min_value=1, max_value=1000, value=10))

    latent_dim = int(
        st.number_input("Latent size", min_value=1, max_value=100, value=10)
    )
    frame = configure_screen_frame()
    return VAETrainingConfig(
        lr=lr,
        observation_frame=frame,
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=64,
        kld_coeff=0.5,
        input_shape=(600, 600),
        validation_fraction=0.2,
    )
