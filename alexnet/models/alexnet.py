import torch
import torch.nn as nn
from omegaconf import DictConfig


class AlexNet(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        c: DictConfig = cfg.model

        cfg_lrn = c.local_response_norm
        self.lrn = nn.LocalResponseNorm(
            cfg_lrn.size, k=cfg_lrn.k, alpha=cfg_lrn.alpha, beta=cfg_lrn.beta
        )
        self.pool = self._get_pool(c.pool)
        self.activation = self._get_activation(c.activation)
        self.dropout = nn.Dropout(c.dropout)

        # Feature Extractors
        self.conv1 = self._make_conv(c.in_channels, c.conv1)
        self.conv2 = self._make_conv(c.conv1.out_channels, c.conv2)
        self.conv3 = self._make_conv(c.conv2.out_channels, c.conv3)
        self.conv4 = self._make_conv(c.conv3.out_channels, c.conv4)
        self.conv5 = self._make_conv(c.conv4.out_channels, c.conv5)

        self.shape_norm = nn.AdaptiveAvgPool2d((c.shape_norm, c.shape_norm))

        # Classifier
        flattened_dim = c.conv5.out_channels * c.shape_norm * c.shape_norm
        self.fc6 = nn.Linear(flattened_dim, c.fc6_out)
        self.fc7 = nn.Linear(c.fc6_out, c.fc7_out)
        self.fc8 = nn.Linear(c.fc7_out, c.out_classes)

    def _make_conv(self, in_channels: int, cfg_block: DictConfig) -> nn.Conv2d:
        return nn.Conv2d(
            in_channels,
            cfg_block.out_channels,
            kernel_size=cfg_block.kernel_size,
            stride=cfg_block.stride,
            padding=cfg_block.padding,
        )

    def _get_pool(self, cfg: DictConfig) -> nn.Module:
        pool_dict = {
            "max": nn.MaxPool2d,
            "average": nn.AvgPool2d,
        }
        if cfg.type not in pool_dict:
            raise ValueError(f"Unsupported pooling operation: {cfg.type}")
        pool_cls = pool_dict[cfg.type]
        return pool_cls(
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            padding=cfg.padding,
        )

    def _get_activation(self, name: str) -> nn.Module:
        activation_dict = {
            "relu": (nn.ReLU, True),
            "gelu": (nn.GELU, False),
            "leaky_relu": (nn.LeakyReLU, True),
        }
        if name not in activation_dict:
            raise ValueError(f"Unsupported activation: {name}")
        cls, supports_inplace = activation_dict[name]
        return cls(inplace=True) if supports_inplace else cls()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature Extraction
        x = self.pool(self.lrn(self.activation(self.conv1(x))))
        x = self.pool(self.lrn(self.activation(self.conv2(x))))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.pool(self.activation(self.conv5(x)))

        # Shaping
        x = self.shape_norm(x)
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.dropout(self.activation(self.fc6(x)))
        x = self.dropout(self.activation(self.fc7(x)))
        x = self.fc8(x)
        return x
