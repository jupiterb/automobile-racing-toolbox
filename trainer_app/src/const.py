from pathlib import Path
from pydantic import BaseConfig
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).absolute().parents[1]
RESOURCE_DIR = ROOT_DIR / "resources"


class EnvVarsConfig(BaseConfig):
    celery_broker_url: str
    celery_backend_url: str


load_dotenv(RESOURCE_DIR / ".env")
