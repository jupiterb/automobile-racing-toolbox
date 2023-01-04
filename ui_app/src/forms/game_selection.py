import streamlit as st
from typing import Optional, Union
from racing_toolbox.interface.models.keyboard_action import KeyAction
import string

from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.interface.models.gamepad_action import (
    GamepadButton,
    GamepadControl,
    GamepadAction,
)
from racing_toolbox.observation.utils.ocr import OcrConfiguration, OcrToolConfiguration
from racing_toolbox.observation.utils import ScreenFrame

from ui_app.src.forms.common import configure_screen_frame
from ui_app.src.shared import Shared


def configure_game() -> Optional[GameConfiguration]:
    source = Shared().games_source
    games = source.get_configs()
    selected = st.selectbox("Select game", list(games.keys()))
    with st.expander("Add new game"):
        name = st.text_input("Provide new game configuration name", value="My_New_Game")
        with st.form("new_game"):
            new_game = create_new_game_config(name)
            submitted = st.form_submit_button("Submit")
            if submitted:
                source.add_config(name, new_game)
    if selected is not None:
        config = games[selected]
        return config


def create_new_game_config(name: str) -> GameConfiguration:
    process_name = st.text_input("Game process name", value="Game process name")
    discrete_actions = configure_discrete_actions()
    continous_actions = configure_continous_actions()
    reset_keys_sequence = configure_discrete_reset_sequence()
    reset_gamepad_sequence = configure_continous_reset_sequence()
    reset_seconds = st.number_input(
        "Number of seconds of delay after apllying rest sequence",
        min_value=0,
        max_value=10,
        value=3,
    )
    frequency = st.number_input(
        "Action -> Observation frequency per second",
        min_value=1,
        max_value=32,
        value=8,
    )
    window_size = (
        st.number_input("Window height", min_value=200, max_value=1600, value=800),
        st.number_input("Window width", min_value=200, max_value=1600, value=1000),
    )

    st.markdown("""---""")
    ocrs = configure_ocr()

    return GameConfiguration(
        game_id=name,
        process_name=process_name,
        window_size=window_size,
        discrete_actions_mapping=discrete_actions,
        continous_actions_mapping=continous_actions,
        reset_keys_sequence=reset_keys_sequence,
        reset_gamepad_sequence=reset_gamepad_sequence,
        reset_seconds=reset_seconds,
        frequency_per_second=frequency,
        ocrs=ocrs,
    )


def get_possible_key_actions() -> list[str]:
    possible_keys: list[Union[KeyAction, str]] = [key.value for key in KeyAction]
    possible_keys.extend(string.ascii_uppercase)
    possible_keys.extend(string.digits)
    return possible_keys


def get_possible_gamepad_actions() -> list[GamepadAction]:
    return [action for _set in [GamepadButton, GamepadControl] for action in _set]


def configure_discrete_actions() -> dict[str, str]:
    possible_actions = get_possible_key_actions()
    keys_to_use = st.multiselect(
        "Choose keyboard actions for discrete action space", possible_actions
    )
    return {str(selected): selected for selected in keys_to_use}


def configure_continous_actions() -> dict[str, GamepadAction]:
    possible_actions = get_possible_gamepad_actions()
    actions_to_use = st.multiselect(
        "Choose gamepad actions for continous action space", possible_actions
    )
    return {str(selected): selected for selected in actions_to_use}


def configure_discrete_reset_sequence() -> list[str]:
    possible_actions = get_possible_key_actions()
    return st.multiselect(
        "Choose keyboard actions for episode resetting", possible_actions
    )


def configure_continous_reset_sequence() -> list[GamepadAction]:
    possible_actions = get_possible_gamepad_actions()
    return st.multiselect(
        "Choose gamepad actions for episode resetting", possible_actions
    )


def configure_ocr() -> OcrToolConfiguration:
    ocr_frame = configure_screen_frame("OCR")
    return OcrToolConfiguration(
        instances={
            "speed": (
                ocr_frame,
                OcrConfiguration(
                    threshold=190,
                    max_digits=3,
                    segemnts_definitions={
                        0: ScreenFrame(top=0, bottom=0.09, left=0.42, right=0.60),
                        1: ScreenFrame(top=0.15, bottom=0.28, left=0.14, right=0.28),
                        2: ScreenFrame(top=0.15, bottom=0.28, left=0.85, right=1.0),
                        3: ScreenFrame(top=0.38, bottom=0.5, left=0.42, right=0.60),
                        4: ScreenFrame(top=0.58, bottom=0.73, left=0.14, right=0.28),
                        5: ScreenFrame(top=0.58, bottom=0.73, left=0.85, right=1.0),
                        6: ScreenFrame(top=0.82, bottom=0.94, left=0.42, right=0.60),
                    },
                ),
            )
        },
    )
