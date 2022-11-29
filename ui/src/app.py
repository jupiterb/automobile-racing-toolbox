from multiprocessing import Process
from pydantic import ValidationError
import streamlit as st
import json
import ipaddress

from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.training.config.user_defined import TrainingConfig

from ui.src.state_utils import (
    SharedState,
    Worker,
    WorkerFailure,
    start_trainer_process,
)
from ui.src.authentication import auth_pannel


SHARED = SharedState()  # proxy for session state to get shared variables between panels


def config_panel() -> bool:
    """loads game, training, env configs to session state. Returns True if its ready, False otherwise"""
    # TODO: make configs disappear from session_state after file was removed
    with st.sidebar.header("Configuration"):
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
            SHARED.game_config = game_config
        except ValidationError as e:
            st.error(f"Invalid game configuration. {e}")
    else:
        st.error("Please select game config")

    if env_data:
        try:
            env_config = EnvConfig(**json.load(env_data))
            SHARED.env_config = env_config
        except ValidationError as e:
            st.error(f"Invalid env configuration. {e}")
    else:
        st.error("Please select env config")

    if training_data:
        try:
            training_config = TrainingConfig(**json.load(training_data))
            SHARED.training_config = training_config
        except ValidationError as e:
            st.error(f"Invalid training configuration. {e}")
    else:
        st.error("Please select training config")

    return SHARED.configs_ready


def trainer_panel() -> bool:
    """Sets up trainer address. Returns True if all data was submitted, False otherwise"""
    st.write("Trainer")
    address = st.text_input(
        "trainer host",
        key=f"trainer_host",
        value="0.0.0.0",
    )
    port = st.number_input(
        "trainer port",
        key=f"trainer_port",
        min_value=8000,
        max_value=8100,
        step=1,
        value=8000,
    )
    if address and port:
        try:
            ip = ipaddress.ip_address(address)
            SHARED.trainer_address = f"{ip}:{port}"
        except ValueError:
            st.error("IP address {address} is not valid")

    return SHARED.trainer_ready


def workers_panel() -> bool:
    """fetch workers data from user, and sync configuration between trainer and workers.
    Returns True if all workers were synced, False otherwise"""

    def get_address(c, i):
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
            value=8000,
            disabled=st.session_state[f"address{i}_disabled"],
        )
        if address and port:
            try:
                ip = ipaddress.ip_address(address)
                return f"{ip}:{port}"
            except ValueError:
                c.error(f"IP address {address} is not valid")

    # setup shared state variables
    for i in range(st.session_state["training_config"].num_rollout_workers):
        if f"address{i}_disabled" not in st.session_state:
            st.session_state[f"address{i}_disabled"] = False
        if f"worker{i}_synced" not in st.session_state:
            st.session_state[f"worker{i}_synced"] = False
        if f"disable_sync{i}" not in st.session_state:
            st.session_state[f"disable_sync{i}"] = True

    cols = st.columns(st.session_state["training_config"].num_rollout_workers)
    for i, c in enumerate(cols):
        c.write(f"Worker {i}")
        if address := get_address(c, i):
            if i not in SHARED.workerset._workers:
                SHARED.workerset.add(Worker(address), i)
            st.session_state[f"disable_sync{i}"] = False
            st.session_state[f"address{i}_disabled"] = True

        if not st.session_state[f"disable_sync{i}"]:
            if (
                c.button(
                    f"sync with {i}",
                    disabled=st.session_state[f"disable_sync{i}"],
                )
                and address
            ):
                st.session_state[f"address{i}_disabled"] = True
                st.session_state[f"disable_sync{i}"] = True
                try:
                    SHARED.workerset[i].sync(
                        SHARED.game_config, SHARED.env_config, SHARED.trainer_address
                    )
                    st.session_state[f"worker{i}_synced"] = True
                    c.write("Synced succesfully")
                except WorkerFailure as exc:
                    c.error(f"{exc.worker_address}: {exc.reason}")
                    st.session_state[f"address{i}_disabled"] = False
                    st.session_state[f"disable_sync{i}"] = False

    return (
        all(w.synced for w in SHARED.workerset.workers)
        and len(SHARED.workerset) == SHARED.training_config.num_rollout_workers
    )


def training_panel():
    def run_trainer():
        st.session_state["training_running"] = True
        process_args = (
            SHARED.game_config,
            SHARED.env_config,
            SHARED.training_config,
            str(st.session_state["trainer_host"]),
            int(st.session_state["trainer_port"]),
        )
        trainer_process = Process(
            target=start_trainer_process,
            args=process_args,
        )
        st.session_state["trainer_process"] = trainer_process
        trainer_process.start()

    def stop_trainer():
        st.session_state["training_running"] = False
        st.session_state["trainer_process"].kill()
        st.session_state["trainer_process"].join()

    # initialize shared state attributes
    if "training_running" not in st.session_state:
        st.session_state["training_running"] = False
    if "trainer_process" not in st.session_state:
        st.session_state["trainer_process"] = None

    if st.button(
        "run_training",
        disabled=st.session_state["training_running"],
    ):
        run_trainer()
        for w in SHARED.workerset.workers:
            w.start()

    if st.button("stop training", disabled=not st.session_state["training_running"]):
        stop_trainer()
        for w in SHARED.workerset.workers:
            try:
                w.stop()
            except WorkerFailure as f:
                st.error(f"{f.worker_address}: {f.reason}: {f.details}")

    return st.session_state["training_running"]


def main():
    configs_ready = config_panel()
    trainer_ready = trainer_panel() if configs_ready else False
    workers_ready = workers_panel() if trainer_ready else False
    training_ready = training_panel() if workers_ready and trainer_ready else False
    # "session:", st.session_state


if __name__ == "__main__":
    st.set_page_config(page_title="Automobile training starter", layout="wide")
    if user := auth_pannel():
        main()
