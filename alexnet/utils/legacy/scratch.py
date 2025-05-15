import torch
import torch.nn as nn
import torch.nn.functional as F
from data_test import train_loader, test_loader
from tqdm import tqdm


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.lrn = nn.LocalResponseNorm(5, k=2, alpha=1e-4, beta=0.75)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)

        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 1000)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(self.lrn(F.relu(self.conv1(x))))
        x = self.lrn(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # linear section
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x


class CIFAR10AlexNet(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(96)
        # self.lrn = nn.LocalResponseNorm(5, k=2, alpha=1e-4, beta=0.75)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(384)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc6 = nn.Linear(256 * 6 * 6, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        self.fc8 = nn.Linear(1024, classes)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(self.lrn(F.relu(self.conv1(x))))
        x = self.lrn(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.avg_pool(self.pool(x))
        x = x.view(x.size(0), -1)

        # linear section
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x


def alexnet_gaussian_init(m):
    """We initialized the weights in each layer from a zero-mean Gaussian distribution with standard deviation 0.01."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


model = CIFAR10AlexNet()
model.apply(alexnet_gaussian_init)
# From the paper
# We initialized the neuron biases in the second, fourth, and ﬁfth convolutional layers, as well as in the fully-connected hidden layers, with the constant 1.
model.conv2.bias.data.fill_(1)
model.conv4.bias.data.fill_(1)
model.conv5.bias.data.fill_(1)
model.fc6.bias.data.fill_(1)
model.fc7.bias.data.fill_(1)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
# optim = torch.optim.Adam(model.parameters(), lr=0.01)

# The heuristic which we followed was to divide the learning rate by 10 when the validation error rate stopped improving with the current learning rate.


def train(model: nn.Module, optimizer, loss_fn, epochs, train_loader, val_loader):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    for epoch in range(epochs):
        local_loss = 0
        accumulation = 0
        for x, y in tqdm(train_loader):
            # Will add in device mapping after done testing.
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accumulation += 1
            local_loss += loss.item()
        # model.eval()
        # with torch.no_grad():
        #     for x, y in tqdm(val_loader):
        #         val_output = model(x)
        #         val_loss = loss_fn(val_output, y)
        # print(f"Epoch {epoch + 1}, Train Loss: {local_loss/accumulation:.4f}, Val Loss, {val_loss.item():.4f}")

        # scheduler.step(val_loss.item())
        # model.train()
    return model


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            logits = model(x)
            _, pred = torch.max(logits, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    model.train()
    print(f"Test Accuracy: {100 * correct // total}%")


if __name__ == "__main__":
    # evaluate(model, test_loader)
    model = train(model, optim, loss_fn, 1, train_loader, test_loader)
    x, y = next(iter(test_loader))
    logits = model(x)
    pred = logits.argmax(dim=1)

    print("Predictions:", pred[:10].tolist())
    print("Targets:", y[:10].tolist())
    print("Unique preds:", pred.unique())
    evaluate(model, test_loader)
