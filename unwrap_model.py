import argparse
import torch
from omegaconf import OmegaConf
from hydra import initialize, compose

from alexnet.models.alexnet import AlexNet
from alexnet.training.classifier import ClassifierTrainingWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="alexnet/configs")
    parser.add_argument("--config-name", type=str, default="config")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    with initialize(version_base=None, config_path=args.config_path):
        cfg = compose(config_name=args.config_name)

    wrapped_model = AlexNet(cfg)
    loss_fn = torch.nn.CrossEntropyLoss()
    lightning_model = ClassifierTrainingWrapper.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        classifier=wrapped_model,
        loss_fn=loss_fn,
        optimizer_cfg=cfg.optimizer,
    )

    unwrapped_model = lightning_model.classifier
    torch.save(unwrapped_model.state_dict(), args.output_path)
