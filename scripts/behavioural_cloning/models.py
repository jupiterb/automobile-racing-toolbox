from torch import nn
import torch as th
import torch.nn.functional as F


device = th.device("cuda" if th.cuda.is_available() else "cpu")



def training_step(model, batch: tuple):
    images, labels = batch
    out = model(images)
    loss = F.binary_cross_entropy(out, labels)
    return loss

def validation_step(model, batch):
    images, labels = batch
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    return {"val_loss": loss.detach()}

def validation_epoch_end(model, outputs):
    batch_losses = [x["val_loss"] for x in outputs]
    epoch_loss = th.stack(batch_losses).mean()  # Combine losses
    return {"val_loss": epoch_loss.item()}

def epoch_end(model, epoch, result):
    print(
        "Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result["train_loss"], result["val_loss"], result["val_acc"]
        )
    )


class MultiLabelnNN(nn.Module):
    def __init__(self, n_channels, n_outputs, w, h):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.classifier = nn.Sequential(
            nn.Linear(linear_input_size, n_outputs), nn.Sigmoid()
        )

    def forward(self, x):
        x = x.to(device)
        x = self.feature_extractor(x)
        return self.classifier(x)
