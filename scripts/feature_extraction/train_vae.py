import os 
from racing_toolbox.observation.config import vae_config
from racing_toolbox.observation.vae.models import VAE
import torch.utils.data as th_data
from torchvision import transforms
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl
import orjson
import wandb
import numpy as np 
from typing import Callable, Generator

from racing_toolbox.datatool.container import DatasetContainer
import torch as th 
from racing_toolbox.datatool.datasets import Dataset, DatasetModel
from contextlib import contextmanager
import tables as tb 

class MemoryDataset(Dataset):
    def __init__(self, path: str):
        self._path = path 

    @contextmanager
    def get(self) -> Generator[DatasetModel, None, None]:
        if not os.path.exists(self._path):
            raise ValueError(f"{self._path} not found!")
        with tb.File(self._path, driver="H5FD_CORE") as file:
            yield DatasetModel(
                fps=int(file.root.fps[0]),
                observations=file.root.observations,
                actions=file.root.actions,
            )

def tensordataset_from_file(
    recordings_refs: list[str],
    transforms: Callable[[np.ndarray], th.Tensor],
) -> th_data.TensorDataset:
    container = DatasetContainer()
    for file_ref in recordings_refs:
        dataset = MemoryDataset(file_ref)
        assert container.try_add(
            dataset
        ), f"Dataset {file_ref} is incompatible with the rest"

    observations: list[th.Tensor] = []
    for observation, _ in container.get_all():
        observations.append(transforms(observation))
    
    return th_data.TensorDataset(th.stack(observations))

def start_vae_training(
    run_name,
    training_params: vae_config.VAETrainingConfig,
    encoder_config: vae_config.VAEModelConfig,
    recordings_refs: list[str],
):
    transform = transforms.Compose(
        [
            lambda i: np.array(i, dtype=np.uint8),
            lambda i: training_params.observation_frame.apply(i),
            transforms.ToTensor(),
            transforms.Resize(training_params.input_shape),
        ]
    )
    dataset = tensordataset_from_file(
        recordings_refs,
        transform,
    )
    val_len = int(training_params.validation_fraction * len(dataset))
    trainset, testset = th_data.random_split(dataset, [len(dataset) - val_len, val_len])
    trainloader = th_data.DataLoader(trainset, batch_size=training_params.batch_size)
    testloader = th_data.DataLoader(testset, batch_size=training_params.batch_size)
    print(f"dtype: {dataset[0][0].dtype}")

    # build encoder / decoder based on configs
    params_dict = (
        orjson.loads(training_params.json())
        | orjson.loads(encoder_config.json())
        | {"in_channels": 3}
    )
    pl_model = VAE(params_dict)
    wandb.init(project="ART", name=f"vae_{run_name}")
    try:
        wandb_logger = WandbLogger(project="ART", log_model="all")
        trainer = pl.Trainer(
            logger=wandb_logger, max_epochs=training_params.epochs, log_every_n_steps=1, accelerator="cpu"
        )
        trainer.fit(
            model=pl_model, train_dataloaders=trainloader, val_dataloaders=testloader
        )
    except Exception:
        raise 
    finally:
        wandb.finish()


def main():
    config = vae_config.VAETrainingConfig(
        lr=0.01,
        epochs=20,
        kld_coeff=0.0001,
        latent_dim=8,
        input_shape=(128, 128),
        validation_fraction=0.1,
        batch_size=64,
        observation_frame=vae_config.ScreenFrame(
            **{
            "top": 0.475,
            "bottom": 0.9125,
            "left": 0.01,
            "right": 0.99
            }
        )
    )
    filters = [
        (16, [3, 3], 2),
        (32, [3, 3], 2),
        (64, [5, 5], 3),
        (128, [5, 5], 3),
    ]
    model_config = vae_config.VAEModelConfig(conv_filters=filters)
    params = config.dict() | model_config.dict() | {"in_channels": 3}
    print(params)

    refs = [
        "C:\\Users\\yogam\\projects\\automobile-racing-toolbox\\recordings\\trackmania\\maciek\\vae1.h5"
    ]
    start_vae_training("test", config, model_config, refs)

if __name__ == "__main__":
    main()