import torch

torch.set_float32_matmul_precision("high")

from alexnet.models.cifar10_alexnet import CIFAR10AlexNet
from alexnet.data.dataset import create_cifar10_dataloader_from_config
from alexnet.training.classifier import ClassifierTrainingWrapper
import torch.nn as nn
from omegaconf import OmegaConf
from pathlib import Path
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning import seed_everything




def load_config(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return OmegaConf.load(path)


config = load_config("/home/cameronolson/documents/AlexNet/alexnet/configs/config.yaml")
seed_everything(config.seed)

wandb_logger = WandbLogger(
    project="alexnet-cifar10",
    name="exp-001",
    config=OmegaConf.to_container(config, resolve=True),
)

train_loader = create_cifar10_dataloader_from_config(config.dataset, train=True)
test_loader = create_cifar10_dataloader_from_config(config.dataset, train=False)
model = CIFAR10AlexNet(config)
loss_fn = nn.CrossEntropyLoss()
model = ClassifierTrainingWrapper(model, loss_fn, config.optimizer.lr)
trainer = L.Trainer(**config.trainer, logger=wandb_logger)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
