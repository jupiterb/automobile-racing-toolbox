from src.tasks import utils
from src.const import EnvVarsConfig
from racing_toolbox.observation.vae.models import VAE
from racing_toolbox.observation.config import vae_config

from celery.utils.log import get_task_logger
from celery.signals import setup_logging
from celery.contrib.abortable import AbortableTask

import torch.utils.data as th_data
from torchvision import transforms
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl
import orjson
import logging
import wandb

logging.basicConfig(
    format="%(levelname)-8s: %(asctime)s - %(name)s.%(funcName)s() - %(message)s"
)

logger = get_task_logger(__name__)

# app = utils.make_celery(EnvVarsConfig(), 1, "offline_tasks")


# @setup_logging.connect
# def setup_celery_logging(**kwargs):
#     pass


# @app.task(bind=True, base=AbortableTask)
# def start_vae_training(
#     self,
#     training_params: vae_config.VAETrainingConfig,
#     encoder_config: vae_config.VAEModelConfig,
#     bucket_name: str,
#     recordings_refs: list[str],
# ):
#     env_vars = EnvVarsConfig()
#     transform = transforms.Compose(
#         [
#             lambda i: training_params.observation_frame.apply(i),
#             transforms.ToTensor(),
#             transforms.Resize(training_params.input_shape),
#         ]
#     )
#     dataset = utils.tensordataset_from_bucket(
#         recordings_refs,
#         bucket_name,
#         env_vars.aws_key,
#         env_vars.aws_secret_key,
#         transform,
#     )
#     val_len = int(training_params.validation_fraction * len(dataset))
#     trainset, testset = th_data.random_split(dataset, [len(dataset) - val_len, val_len])
#     trainloader = th_data.DataLoader(trainset, batch_size=training_params.batch_size)
#     testloader = th_data.DataLoader(testset, batch_size=training_params.batch_size)

#     # build encoder / decoder based on configs
#     params_dict = (
#         orjson.loads(training_params.json())
#         | orjson.loads(encoder_config.json())
#         | {"in_channels": 3}
#     )
#     pl_model = VAE(params_dict)
#     wandb_logger = WandbLogger(project="ART", log_model="all")
#     trainer = pl.Trainer(
#         logger=wandb_logger, max_epochs=150, log_every_n_steps=1, accelerator="gpu"
#     )

#     with wandb.init(project="ART") as run:
#         trainer.fit(
#             model=pl_model, train_dataloaders=trainloader, val_dataloaders=testloader
#         )
#     wandb.finish()
