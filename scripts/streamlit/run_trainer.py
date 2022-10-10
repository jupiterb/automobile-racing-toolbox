from cgitb import enable
import functools
from pydantic import ValidationError
import streamlit_pydantic as sp
import streamlit as st
import json
import ipaddress
import random

from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.training import Trainer, config
from racing_toolbox.training.config.user_defined import TrainingConfig


def get_files():
    with st.sidebar.header("Source data selection"):
        st.sidebar.write("Game Configuration")
        game_data = st.sidebar.file_uploader(
            "Upload/select game config JSON data", type=["JSON"]
        )
        st.sidebar.write("Training Configuration")
        training_data = st.sidebar.file_uploader(
            "Upload/select training config JSON data", type=["JSON"]
        )
        st.sidebar.write("Enviroment Configuration")
        env_data = st.sidebar.file_uploader(
            "Upload/select enviroment config JSON data", type=["JSON"]
        )

    if game_data:
        try:
            game_config = GameConfiguration(**json.load(game_data))
            if "game_conf" not in st.session_state:
                st.session_state["game_conf"] = game_config
        except ValidationError as e:
            st.error(f"Invalid game configuration. {e}")
    else:
        st.error("Please select game config")

    if env_data:
        try:
            env_config = EnvConfig(**json.load(env_data))
            if "env_config" not in st.session_state:
                st.session_state["env_config"] = env_config
        except ValidationError as e:
            st.error(f"Invalid env configuration. {e}")
    else:
        st.error("Please select env config")

    if training_data:
        try:
            training_config = TrainingConfig(**json.load(training_data))
            if "training_config" not in st.session_state:
                st.session_state["training_config"] = training_config
        except ValidationError as e:
            st.error(f"Invalid training configuration. {e}")
    else:
        st.error("Please select training config")


def syn_with_worker(c, i):
    """
    Disable worker address input from changing, send request to worker,
    if not successfull show error msg and enable input
    """
    st.session_state[f"address{i}_disabled"] = True
    st.session_state[f"disable_sync{i}"] = True
    if random.random() < 0.5:
        st.session_state[f"address{i}_disabled"] = False
        st.session_state[f"disable_sync{i}"] = False
        c.write("unsuccessfull sync")
    else:
        c.write("Synced succesfully")


def setup_workers(cols):
    def get_address(c, i):
        if f"address{i}_disabled" not in st.session_state:
            st.session_state[f"address{i}_disabled"] = False

        address = c.text_input(
            "Worker ip address",
            key=f"ip{i}",
            disabled=st.session_state[f"address{i}_disabled"],
        )
        port = c.number_input(
            "Worker port number",
            key=f"port{i}",
            min_value=8000,
            max_value=8100,
            step=1,
            value=8080,
            disabled=st.session_state[f"address{i}_disabled"],
        )
        if address and port:
            try:
                ip = ipaddress.ip_address(address)
                setattr(st.session_state, f"address{i}", f"{ip}:{port}")
            except ValueError:
                c.error(f"IP address {address} is not valid")

    for i, c in enumerate(cols):
        get_address(c, i)
        if f"address{i}" in st.session_state:
            if f"disable_sync{i}" not in st.session_state:
                st.session_state[f"disable_sync{i}"] = False
            clbck = functools.partial(syn_with_worker, c, i)
            c.button(
                f"sync with {i}",
                on_click=clbck,
                disabled=st.session_state[f"disable_sync{i}"],
            )


def main():
    get_files()
    if "training_config" in st.session_state:
        training_config = st.session_state["training_config"]
        cols = st.columns(training_config.num_rollout_workers)
        for i, c in enumerate(cols):
            c.write(f"Worker {i}")
        if "env_config" in st.session_state:
            setup_workers(cols)
        "session:", st.session_state


if __name__ == "__main__":
    st.set_page_config(page_title="Automobile training starter", layout="wide")
    main()
