from pathlib import Path
from pydantic import BaseSettings, validator
from dotenv import load_dotenv
from racing_toolbox.logger import setup_logger
import socket 
import logging 


ROOT_DIR = Path(__file__).absolute().parents[1]
RESOURCE_DIR = ROOT_DIR / "resources"
TMP_DIR = ROOT_DIR / ".tmp"
TMP_DIR.mkdir(exist_ok=True)
load_dotenv(RESOURCE_DIR / ".env")
setup_logger(RESOURCE_DIR / "logger.yml")


class EnvVarsConfig(BaseSettings):
    celery_broker_url: str 
    celery_backend_url: str 
    default_policy_port: int
    default_policy_host: str

logging.warning(f"Following enviroment variables were setup: {EnvVarsConfig()}")
