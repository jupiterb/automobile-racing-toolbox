from fastapi import APIRouter
from src.schemas import StartVAETrainingRequest
from src.tasks import offline_tasks
from logging import getLogger

logger = getLogger(__name__)

offline_router = APIRouter(prefix="/offline")


@offline_router.post("/start_vae")
def start_vae_training_task(body: StartVAETrainingRequest):
    task = offline_tasks.start_vae_training.delay(
        training_params=body.training_config,
        encoder_config=body.encoder_config,
        bucket_name=body.bucket_name,
        recordings_refs=body.recordings_refs
    )
    return task.id
