import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from hydra import initialize, compose
from omegaconf import OmegaConf

from alexnet.models.alexnet import AlexNet
from alexnet.data.dataset import create_cifar10_dataloader_from_config


def evaluate(model: torch.nn.Module, test_loader: DataLoader, device: torch.device):
    print("Testing...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, pred = torch.max(logits, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    model.train()
    print(f"Test Accuracy: {100 * correct // total}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="alexnet/configs")
    parser.add_argument("--config-name", type=str, default="config")
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()

    with initialize(version_base=None, config_path=args.config_path):
        cfg = compose(config_name=args.config_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlexNet(cfg)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # Will make this configurable to the dataset I am training on
    test_loader = create_cifar10_dataloader_from_config(cfg.dataset, train=False)
    evaluate(model, test_loader, device)
    