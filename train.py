from model import CIFAR10AlexNet
from utils import overfit_single_batch, train, evaluate
import torch
import torch.nn as nn
from data_test import train_loader

model = CIFAR10AlexNet()
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# overfit_single_batch(model, loss_fn, optim, train_loader, epochs=20)
model = train(model, optim, loss_fn, 10, train_loader, device)
# evaluate