from pydantic import BaseSettings
from dotenv import load_dotenv
from pathlib import Path
import logging
from racing_toolbox.logger import setup_logger


ROOT_DIR = Path(__file__).absolute().parents[1]
RESOURCE_DIR = ROOT_DIR / "resources"
load_dotenv(RESOURCE_DIR / ".env")
setup_logger(RESOURCE_DIR / "logger.yml")


class EnvVarsConfig(BaseSettings):
    aws_key_id: str
    aws_secret_key: str


logging.info(f"Following enviroment variables were setup: {EnvVarsConfig()}")
