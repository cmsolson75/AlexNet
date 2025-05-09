import torchvision
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=ToTensor())


train_loader = DataLoader(train_data, batch_size=128, shuffle=True)