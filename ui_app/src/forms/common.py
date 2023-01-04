import streamlit as st
import json
from typing import get_args


from racing_toolbox.observation.utils.screen_frame import ScreenFrame
from racing_toolbox.training.config import ModelConfig
from racing_toolbox.training.config.user_defined import Activation

from ui_app.src.shared import Shared
from ui_app.src.utils import SetEncoder


def configure_screen_frame(of_what: str = "screen") -> ScreenFrame:
    st.write(f"Configure {of_what} frame")
    try:
        return ScreenFrame(
            top=st.number_input(
                "Top", min_value=0.0, max_value=1.0, value=0.0, step=0.05
            ),
            bottom=st.number_input(
                "Bottom", min_value=0.0, max_value=1.0, value=1.0, step=0.05
            ),
            left=st.number_input(
                "Left", min_value=0.0, max_value=1.0, value=0.0, step=0.05
            ),
            right=st.number_input(
                "Right", min_value=0.0, max_value=1.0, value=1.0, step=0.05
            ),
        )
    except Exception as e:
        st.warning(f"Provided values are unvalid: {e}")
        st.warning("Default frame will be used")
        return ScreenFrame()


def select_recordings() -> list[str]:
    """returns references of selected recordings"""
    try:
        source = Shared().recordings_source
        recordings_to_use = source.get_recordings()
    except:
        # There is no recordings source
        st.warning("There is no recordings source.")
        recordings_to_use = {}
    selected = st.multiselect(
        "Select recordings to use in training",
        list(recordings_to_use.keys()),
    )
    st.write("You can upload recordings in side panel")
    upload_recording()
    return [recordings_to_use[name] for name in selected]


def upload_recording():
    st.sidebar.markdown("""---""")
    st.sidebar.header("Upload recordings")
    st.sidebar.markdown("""---""")
    shared = Shared()
    source = shared.recordings_source
    uploaded = st.sidebar.file_uploader("Upload new recording", type=["H5"])
    if uploaded:
        if st.sidebar.button("Upload"):
            source.upload_recording(uploaded.name, uploaded)


def configure_model() -> ModelConfig:
    st.markdown("""---""")
    model = ModelConfig(
        fcnet_hiddens=[100, 256],
        fcnet_activation="relu",
        conv_filters=[
            (32, (8, 8), 4),
            (64, (4, 4), 2),
            (64, (3, 3), 1),
            (64, (8, 8), 1),
        ],
    )
    model_str = st.text_area(
        "Configure model architecture",
        json.dumps(model.dict(), cls=SetEncoder, indent=4),
        height=200,
    )
    avtivations = get_args(Activation)
    st.write(f"Where possible activations are {avtivations}")
    try:
        model = ModelConfig(**json.loads(model_str))
    except:
        st.warning("Oops. Your model is not valid")
    return model
