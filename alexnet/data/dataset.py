import torchvision
from torchvision.transforms import transforms
import torch


def create_cifar10_dataloader_from_config(
    config: dict, train: bool, download: bool = True
) -> torch.utils.data.DataLoader:
    batch_size = config.get("batch_size", None)
    if batch_size is None:
        raise ValueError("Missing batch size")

    root_dir = config.get("data_directory", None)
    if root_dir is None:
        raise ValueError("Missing batch size")

    num_workers = config.get("num_workers", 1)

    if train:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )
        shuffle = True
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )
        shuffle = False

    dataset = torchvision.datasets.CIFAR10(
        root=root_dir, train=train, download=download, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
