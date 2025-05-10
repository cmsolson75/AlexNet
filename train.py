from model import CIFAR10AlexNet, SmallNet
from utils import alexnet_gaussian_init, overfit_single_batch
import torch
import torch.nn as nn
from data_test import train_loader


model = CIFAR10AlexNet()
# model.apply(alexnet_gaussian_init)
# # From the paper
# # We initialized the neuron biases in the second, fourth, and ﬁfth convolutional layers, as well as in the fully-connected hidden layers, with the constant 1.
# model.conv2.bias.data.fill_(1)
# model.conv4.bias.data.fill_(1)
# model.conv5.bias.data.fill_(1)
# model.fc6.bias.data.fill_(1)
# model.fc7.bias.data.fill_(1)

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

overfit_single_batch(model, loss_fn, optim, train_loader, epochs=20)