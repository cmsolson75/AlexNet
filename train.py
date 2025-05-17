import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from alexnet.models.alexnet import AlexNet
from alexnet.data.dataset import create_cifar10_dataloader_from_config
from alexnet.training.classifier import ClassifierTrainingWrapper

torch.set_float32_matmul_precision("high")

@hydra.main(version_base=None, config_path="alexnet/configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    wandb_logger = WandbLogger(
        project="alexnet-cifar10",
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    train_loader = create_cifar10_dataloader_from_config(cfg.dataset, train=True)
    test_loader = create_cifar10_dataloader_from_config(cfg.dataset, train=False)
    model = AlexNet(cfg)
    loss_fn = torch.nn.CrossEntropyLoss()
    model = ClassifierTrainingWrapper(model, loss_fn, cfg.optimizer)
    trainer = L.Trainer(**cfg.trainer, logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

if __name__ == "__main__":
    main()