from pydantic import BaseSettings
from dotenv import load_dotenv
from pathlib import Path
import logging
from racing_toolbox.logger import setup_logger


ROOT_DIR = Path(__file__).absolute().parents[2]
RESOURCE_DIR = ROOT_DIR / "resources"
load_dotenv(RESOURCE_DIR / ".env")
setup_logger(RESOURCE_DIR / "logger.yml")


class EnvVarsConfig(BaseSettings):
    aws_key_id: str
    aws_secret_key: str
    bucket_name: str
    access_group_name: str
    redirect_uri: str
    google_client_id: str
    google_client_secret: str


logging.info(f"Following enviroment variables were setup: {EnvVarsConfig()}")
