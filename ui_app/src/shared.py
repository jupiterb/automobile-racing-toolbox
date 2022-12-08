import streamlit as st
from boto3 import Session

from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.config import EnvConfig
from racing_toolbox.training.config import TrainingConfig

from ui_app.src.config import (
    UIAppConfig,
    ConfigsContentEndpoints,
    ConfigSourceType,
    Credentials,
)
from ui_app.src.config_source.abstract import AbstractConfigSource
from ui_app.src.config_source.file_based import FileSysteConfigSource
from ui_app.src.config_source.s3_bucket import S3BucketConfigSource
from ui_app.src.recordings_source.abstract import AbstractRecordingsScource
from ui_app.src.recordings_source.s3_bucket import S3BucketRecordingsSource
from ui_app.src.utils import UIAppError


class Shared:

    GAME_CONFIGS = "game_configs"
    ENV_CONFIGS = "env_configs"
    TRAINING_CONFIGS = "training_configs"
    S3_SESSION = "s3_session"
    RECORDINGS = "recordings"

    def use_config(self, config: UIAppConfig) -> None:
        if config.source_type == ConfigSourceType.FILE_BASED:
            self._init_file_based_sources(config.root, config.endpoints)
        elif config.source_type == ConfigSourceType.S3_BUCKET:
            if config.credentials is not None:
                self._init_s3_bucket_sources(
                    config.credentials, config.root, config.endpoints
                )
            else:
                raise UIAppError(
                    "Cannot establish connection with S3 without credentials"
                )

    @property
    def games_source(self) -> AbstractConfigSource[GameConfiguration]:
        return st.session_state[Shared.GAME_CONFIGS]

    @property
    def envs_source(self) -> AbstractConfigSource[EnvConfig]:
        return st.session_state[Shared.ENV_CONFIGS]

    @property
    def training_configs_source(self) -> AbstractConfigSource[TrainingConfig]:
        return st.session_state[Shared.TRAINING_CONFIGS]

    @property
    def recordings_source(self) -> AbstractRecordingsScource:
        return st.session_state[Shared.RECORDINGS]

    def _init_file_based_sources(self, root: str, folders: ConfigsContentEndpoints):
        st.session_state[Shared.GAME_CONFIGS] = FileSysteConfigSource[
            GameConfiguration
        ](f"{root}/{folders.game_configs}")

        st.session_state[Shared.ENV_CONFIGS] = FileSysteConfigSource[EnvConfig](
            f"{root}/{folders.env_configs}"
        )

        st.session_state[Shared.TRAINING_CONFIGS] = FileSysteConfigSource[
            TrainingConfig
        ](f"{root}/{folders.training_configs}")

    def _init_s3_bucket_sources(
        self,
        credentials: Credentials,
        bucket_name: str,
        key_prefixes: ConfigsContentEndpoints,
    ):
        session = Session(credentials.user_key_id, credentials.user_secret_key)

        st.session_state[Shared.GAME_CONFIGS] = S3BucketConfigSource[GameConfiguration](
            session, bucket_name, key_prefixes.game_configs
        )

        st.session_state[Shared.ENV_CONFIGS] = S3BucketConfigSource[EnvConfig](
            session, bucket_name, key_prefixes.env_configs
        )

        st.session_state[Shared.TRAINING_CONFIGS] = S3BucketConfigSource[
            TrainingConfig
        ](session, bucket_name, key_prefixes.training_configs)

        st.session_state[Shared.RECORDINGS] = S3BucketRecordingsSource(
            session, bucket_name, key_prefixes.recordings
        )
