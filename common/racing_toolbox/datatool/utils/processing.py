import torch.utils.data as th_data
from torchvision import transforms
import numpy as np 
from typing import Callable
import torch as th 

from racing_toolbox.datatool.container import DatasetContainer
from racing_toolbox.datatool.datasets.from_memory import LocalDataset


def tensordataset_from_file(
    recordings_refs: list[str],
    transform: Callable[[np.ndarray], th.Tensor],
) -> th_data.TensorDataset:
    container = DatasetContainer()
    for file_ref in recordings_refs:
        dataset = LocalDataset(file_ref)
        assert container.try_add(
            dataset
        ), f"Dataset {file_ref} is incompatible with the rest"

    observations: list[th.Tensor] = []
    for observation, _ in container.get_all():
        i = transform(observation)
        i_f = transforms.functional.hflip(i)
        observations.append(i)
        observations.append(i_f)

    
    return th_data.TensorDataset(th.stack(observations))
