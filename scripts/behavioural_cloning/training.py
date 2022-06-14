from pathlib import Path
from torch.utils.data import Dataset, random_split
import torch as th
from torch import nn
import tables
import numpy as np
import logging
from tqdm import tqdm
from dataclasses import dataclass
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader
from models import MultiLabelnNN

logger = logging.getLogger(__name__)


class ExpertDataset(Dataset):
    def __init__(self, hdf5_path: Path):
        hdf5_file = tables.open_file(hdf5_path, mode="r")
        self._images = np.moveaxis(hdf5_file.root.images[:], 3, 1)
        self._actions = hdf5_file.root.actions[:].astype(np.double)
        hdf5_file.close()
        logger.info(
            f"Read dataset of shape {self._images.shape} and actions {self._actions.shape}"
        )

    def __getitem__(self, index):
        return ((self._images[index] / 255).astype(np.double), self._actions[index])

    def __len__(self):
        return len(self._images)


@dataclass
class TrainingParams:
    batch_size: int
    epochs: int
    scheduller_gamma: float
    learning_rate: float
    seed: int


def main():

    dataset = ExpertDataset(Path("/home/czyjtu/private/automobile-racing-toolbox/data/tmnf_1os.hdf5"))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    dataset_train, dataset_val = random_split(dataset, [train_size, test_size])

    train_dl = DataLoader(
        dataset_train, 64, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dl = DataLoader(dataset_val, 64, num_workers=4, pin_memory=True)

    model = MultiLabelnNN(4, 4, 53, 150).double()

    fit(1_000, 0.001, model, train_dl, val_dl)


@th.no_grad()
def evaluate(model, val_loader, loss_fn):
    model.eval()
    outputs = [validation_step(model, batch, loss_fn) for batch in val_loader]
    return validation_epoch_end(model, outputs)


def fit(
    epochs,
    lr,
    model,
    train_loader,
    val_loader,
    opt_func=th.optim.Adam,
    checkpoint_step=100,
    checkpoint_dir: Path = Path("checkpoints"),
):
    loss_fn = F.binary_cross_entropy
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in tqdm(range(epochs)):

        model.train()
        train_losses = []
        for batch in train_loader:
            loss = training_step(model, batch, loss_fn)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader, loss_fn)
        result["train_loss"] = th.stack(train_losses).mean().item()
        epoch_end(model, epoch, result)
        history.append(result)
        if epoch % checkpoint_step == 0:
            th.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "result": result
                },
                checkpoint_dir / f"epoch_{epoch}",
            )

    return history


def training_step(model, batch: tuple, loss_fn):
    images, labels = batch
    out = model(images)
    loss = loss_fn(out, labels)
    return loss


def validation_step(model, batch, loss_fn):
    images, labels = batch
    out = model(images)  # Generate predictions
    loss = loss_fn(out, labels)  # Calculate loss
    return {"val_loss": loss.detach()}


def validation_epoch_end(model, outputs):
    batch_losses = [x["val_loss"] for x in outputs]
    epoch_loss = th.stack(batch_losses).mean()  # Combine losses
    return {"val_loss": epoch_loss.item()}


def epoch_end(model, epoch, result):
    print(
        "Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, result["train_loss"], result["val_loss"]
        )
    )



if __name__ == '__main__':
    main()