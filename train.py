import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from alexnet.models.alexnet import AlexNet
from alexnet.data.factory import create_dataloader_from_config
from alexnet.training.classifier import ClassifierTrainingWrapper

torch.set_float32_matmul_precision("high")


def create_checkpoint_callback(cfg):
    return ModelCheckpoint(
        filename=cfg.checkpoint.filename,
        save_last=cfg.checkpoint.save_last,
        every_n_train_steps=cfg.checkpoint.every_n_train_steps,
        every_n_epochs=cfg.checkpoint.every_n_epochs,
    )


@hydra.main(version_base=None, config_path="alexnet/configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    wandb_logger = WandbLogger(
        project=cfg.project,
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    # Needs to be configurable: use a factory for different datasets
    train_loader = create_dataloader_from_config(cfg.dataset, train=True)
    test_loader = create_dataloader_from_config(cfg.dataset, train=False)

    model = AlexNet(cfg)
    loss_fn = torch.nn.CrossEntropyLoss()
    wrapper = ClassifierTrainingWrapper(model, loss_fn, cfg)

    checkpoint_callback = create_checkpoint_callback(cfg)

    trainer = L.Trainer(
        **cfg.trainer,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=wrapper,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
        ckpt_path=cfg.checkpoint.resume.ckpt_path,
    )


if __name__ == "__main__":
    main()
