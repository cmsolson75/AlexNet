from alexnet.data.dataset import create_cifar100_dataloader_from_config, create_cifar10_dataloader_from_config
from omegaconf import DictConfig


_DATASET_FACTORY = {
    "cifar10": create_cifar10_dataloader_from_config,
    "cifar100": create_cifar100_dataloader_from_config
}


def create_dataloader_from_config(config: DictConfig, train: bool, download: bool = True):
    dataset_name = config.get("dataset_name")
    if dataset_name not in _DATASET_FACTORY:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return _DATASET_FACTORY[dataset_name](config=config, train=train, download=download)