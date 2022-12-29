from pydantic import BaseModel, BaseSettings
from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parents[1]
RESOURCE_DIR = ROOT_DIR / "resources"
load_dotenv(RESOURCE_DIR / ".env")


class UserData(BaseModel):
    email: str
    username: str
    password: str
    user_key_id: str
    user_secret_key: str
    wandb_api_key: str


class AppConfig(BaseSettings):
    trainer_url: str
    registry_url: str
    redirect_url: str
    bucket_name: str
    google_client_id: str
    google_client_secret: str
