import torch.nn as nn
import torch.nn.functional as F


# Original init strategy from AlexNet
def alexnet_gaussian_init(m):
    """We initialized the weights in each layer from a zero-mean Gaussian distribution with standard deviation 0.01."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Should implement this as a CONFIG instead of its own file

# Implementation from
# ImageNet Classiﬁcation with Deep Convolutional Neural Networks
# No configuration control
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
