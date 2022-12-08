import streamlit as st

from racing_toolbox.observation.utils.screen_frame import ScreenFrame

from ui_app.src.shared import Shared


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
    """returns names of selected recordings"""
    try:
        source = Shared().recordings_source
        recordings_to_use = source.get_recordings()
    except:
        # There is no recordings source
        st.warning("There is no recordings source.")
        recordings_to_use = {}
    selected = st.multiselect(
        "Select recordings to use in training", list(recordings_to_use.keys())
    )
    return [recordings_to_use[name] for name in selected]
