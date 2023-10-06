import torch.nn as nn
from torch import optim


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 28 * 28, 10)  # 10 classes for MNIST
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32 * 28 * 28)
        x = self.fc(x)
        return x

    def train(self, adv_outputs, labels):
        self.optimizer.zero_grad()
        d_loss = nn.CrossEntropyLoss()(adv_outputs, labels)
        d_loss.backward()
        self.optimizer.step()
        return d_loss.item()
