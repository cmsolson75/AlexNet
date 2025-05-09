import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor


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

        self.fc6 = nn.Linear(256*6*6, 4096)
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



def alexnet_gaussian_init(m):
    """We initialized the weights in each layer from a zero-mean Gaussian distribution with standard deviation 0.01."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model = AlexNet()
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
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=5)


train_data = torch.randn(128, 3, 224, 224)
train_target = torch.randint(0, 1000, (128,))

val_data = torch.randn(64, 3, 224, 224)
val_target = torch.randint(0, 1000, (64,))


EPOCHS = 20

for i in range(EPOCHS):
    logits = model(train_data)
    loss = loss_fn(logits, train_target)
    optim.zero_grad()
    loss.backward()
    optim.step()

    # model.eval()
    # with torch.no_grad():
    #     val_output = model(val_data)
    #     val_loss = loss_fn(val_output, val_target)
    print(f"Epoch {i + 1}, Train Loss: {loss.item():.4f}")
    
    # scheduler.step(val_loss.item())



def train(model: nn.Module, optimizer, loss_fn, epochs, train_loader, val_loader):
    pass