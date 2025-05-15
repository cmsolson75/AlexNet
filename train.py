from model import CIFAR10AlexNet
from utils import train, evaluate
import torch
import torch.nn as nn
from data_test import train_loader, test_loader
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter()

model = CIFAR10AlexNet()
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = train(model, optim, loss_fn, 40, train_loader, test_loader, device, writer)
evaluate(model, test_loader, device)

writer.close()