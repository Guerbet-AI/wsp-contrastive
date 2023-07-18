import torch.nn as nn
import torch.nn.functional as F
import torch


class TinyNet(nn.Module):

    def __init__(self,
                 pretrained: bool = False,
                 num_classes: int = 2,
                 mode: str = 'encoder',
                 rep_dim: int = 512,
                 hidden_dim: int = 256,
                 output_dim: int = 128):
        super().__init__()
        self.mode = mode
        if pretrained:
            self.conv1 = nn.Conv2d(3, 16, 5)
        else:
            self.conv1 = nn.Conv2d(1, 16, 5)

        self.conv1 = nn.Conv2d(2, 16, 5)
        self.features_conv = nn.Sequential(self.conv1,
                                           nn.ReLU(),
                                           nn.MaxPool2d(2, 2),
                                           nn.Conv2d(16, 32, 5),
                                           nn.ReLU(),
                                           nn.MaxPool2d(2, 2),
                                           nn.Conv2d(32, 64, 5),
                                           nn.ReLU(),
                                           nn.MaxPool2d(2, 2),
                                           nn.Conv2d(64, 128, 5),
                                           nn.ReLU(),
                                           nn.MaxPool2d(2, 2),
                                           nn.Conv2d(256, rep_dim, 5),
                                           nn.ReLU()
                                           )

        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(rep_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.fc_cl = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mode=None):
        if mode is None:
            mode = self.mode
        x = self.features_conv(x)
        x = self.pool(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        if mode == 'representation':
            return x.squeeze(dim=1)
        elif mode == 'encoder':
            x = F.relu(self.fc1(x))
            x = self.fc_cl(x)
        elif mode == 'classifier':
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        return x.squeeze(dim=1)
