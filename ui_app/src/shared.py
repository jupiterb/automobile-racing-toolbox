import streamlit as st
from boto3 import Session
from httpx_oauth.oauth2 import OAuth2Token

from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.config import EnvConfig
from racing_toolbox.training.config import TrainingConfig
from racing_toolbox.observation.config.vae_config import (
    VAETrainingConfig,
    VAEModelConfig,
)

from ui_app.src.config import UserData, AppConfig
from ui_app.src.config_source.abstract import AbstractConfigSource
from ui_app.src.config_source.s3_bucket import S3BucketConfigSource
from ui_app.src.recordings_source.abstract import AbstractRecordingsScource
from ui_app.src.recordings_source.s3_bucket import S3BucketRecordingsSource
from ui_app.src.services import TrainerService, RegistryService


class Shared:

    JUST_LOGGED = "logged"
    TOKEN = "token"
    USER_DATA = "user_data"
    IS_GOOGLE_USER = "is_google_user"
    TRAINER_SERVICE = "trainer_server"
    REGISTRY_SERVICE = "registry_service"
    GAME_CONFIGS = "game_configs"
    ENV_CONFIGS = "env_configs"
    TRAINING_CONFIGS = "training_configs"
    VAE_TRAINING_CONFIGS = "vae_training_configs"
    VAE_MODELS = "vae_models"
    S3_SESSION = "s3_session"
    RECORDINGS = "recordings"

    def __init__(self) -> None:
        app_conifg = AppConfig()

        st.session_state[Shared.TRAINER_SERVICE] = TrainerService(
            app_conifg.trainer_url
        )
        st.session_state[Shared.REGISTRY_SERVICE] = RegistryService(
            app_conifg.registry_url
        )

    @property
    def just_logged(self) -> bool:
        return st.session_state[Shared.JUST_LOGGED]

    @just_logged.setter
    def just_logged(self, logged: bool) -> None:
        st.session_state[Shared.JUST_LOGGED] = logged

    @property
    def token(self) -> OAuth2Token:
        return st.session_state[Shared.TOKEN]

    @token.setter
    def token(self, token: OAuth2Token):
        st.session_state[Shared.TOKEN] = token

    @property
    def user_data(self) -> UserData:
        return st.session_state[Shared.USER_DATA]

    @user_data.setter
    def user_data(self, data: UserData) -> None:
        initialized = Shared.USER_DATA in st.session_state
        st.session_state[Shared.USER_DATA] = data
        if initialized:
            return
        app_conifg = AppConfig()
        self._init_s3_bucket_sources(
            data.username,
            data.user_key_id,
            data.user_secret_key,
            app_conifg.bucket_name,
        )

    @property
    def is_google_user(self) -> bool:
        return st.session_state[Shared.IS_GOOGLE_USER]

    @is_google_user.setter
    def is_google_user(self, value: bool):
        st.session_state[Shared.IS_GOOGLE_USER] = value

    @property
    def trainer_service(self) -> TrainerService:
        return st.session_state[Shared.TRAINER_SERVICE]

    @property
    def registry_service(self) -> RegistryService:
        return st.session_state[Shared.REGISTRY_SERVICE]

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
    def vae_models_source(self) -> AbstractConfigSource[VAEModelConfig]:
        return st.session_state[Shared.VAE_MODELS]

    @property
    def vae_training_configs_source(self) -> AbstractConfigSource[VAETrainingConfig]:
        return st.session_state[Shared.VAE_TRAINING_CONFIGS]

    @property
    def recordings_source(self) -> AbstractRecordingsScource:
        return st.session_state[Shared.RECORDINGS]

    def _init_s3_bucket_sources(
        self, username: str, user_key_id: str, user_secret_key: str, bucket_name: str
    ):
        session = Session(user_key_id, user_secret_key)

        st.session_state[Shared.GAME_CONFIGS] = S3BucketConfigSource[GameConfiguration](
            session, bucket_name, f"users/{username}/configs/games"
        )
        st.session_state[Shared.ENV_CONFIGS] = S3BucketConfigSource[EnvConfig](
            session, bucket_name, f"users/{username}/configs/envs"
        )
        st.session_state[Shared.TRAINING_CONFIGS] = S3BucketConfigSource[
            TrainingConfig
        ](session, bucket_name, f"users/{username}/configs/trainings")
        st.session_state[Shared.VAE_MODELS] = S3BucketConfigSource[VAEModelConfig](
            session, bucket_name, f"users/{username}/configs/vae_models"
        )
        st.session_state[Shared.VAE_TRAINING_CONFIGS] = S3BucketConfigSource[
            VAETrainingConfig
        ](session, bucket_name, f"users/{username}/configs/vae_trainings")

        st.session_state[Shared.RECORDINGS] = S3BucketRecordingsSource(
            session, bucket_name, f"users/{username}/recordings"
        )