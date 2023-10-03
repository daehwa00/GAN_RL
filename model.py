import torch
import torch.nn as nn


# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)

        self.action_mean = nn.Linear(512, 3)  # For x, y, brightness
        self.action_std = nn.Linear(512, 3)  # For x, y, brightness

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(-1, 64 * 28 * 28)
        x = self.fc1(x)
        x = nn.ReLU()(x)

        action_mean = self.action_mean(x)
        action_std = self.action_std(x)
        action_std = nn.Softplus()(action_std)  # Ensure standard deviation is positive

        return action_mean, action_std


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 28 * 28, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32 * 28 * 28)
        x = self.fc(x)
        return x
