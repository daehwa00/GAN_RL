import wandb
from RL_utils import sample_action_and_logprob, add_brightness_to_batch_images
from model import Generator, Discriminator

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
batch_size = 6000
epochs = 50

# WandB – Login to your wandb account so you can log all your metrics
wandb.login()

# Loss Function
criterion = nn.CrossEntropyLoss()

# Early Stopping Parameters
patience = 10  # Number of epochs to wait for improvement

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data loader
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_loader = DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)
test_loader = DataLoader(
    datasets.MNIST("./data", train=False, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)

# Sweep configuration
sweep_config = {
    "name": "My Sweep",
    "method": "bayes",  # Bayesian optimization
    "metric": {
        "goal": "maximize",  # We want to maximize accuracy
        "name": "Test Accuracy",
    },
    "parameters": {
        "g_learning_rate": {"min": 0.00001, "max": 0.001},
        "d_learning_rate": {"min": 0.00001, "max": 0.001},
    },
}


# Define training function
def train():
    sweep_config_defaults = {
        "g_learning_rate": 0.00001,
        "d_learning_rate": 0.00001,
    }

    wandb.init(config=sweep_config_defaults)
    # 각 sweep마다 이름 지정
    run_name = (
        f"g_lr_{wandb.config.g_learning_rate}_d_lr_{wandb.config.d_learning_rate}"
    )
    wandb.run.name = run_name

    best_test_accuracy = 0.0
    counter = 0

    # Instantiate Models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=wandb.config.g_learning_rate)
    d_optimizer = optim.Adam(
        discriminator.parameters(), lr=wandb.config.d_learning_rate
    )

    wandb.watch((generator, discriminator), log="all")

    # Training Loop
    for epoch in range(epochs):
        total_d_loss = 0
        total_g_loss = 0
        correct_labels = 0
        total_labels = 0

        action_map = torch.zeros((28, 28), device=device)
        brightness_values = np.zeros(256)

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            action_mean, action_std = generator(images)

            actions, log_probs = sample_action_and_logprob(action_mean, action_std)
            actions, adv_images = add_brightness_to_batch_images(images, actions)

            # Zero Gradients
            d_optimizer.zero_grad()

            # Forward Pass through Discriminator
            with torch.no_grad():
                real_outputs = discriminator(images)
            adv_outputs = discriminator(adv_images)

            # Calculate Loss
            d_loss = criterion(adv_outputs, labels)

            # Backward Pass
            d_loss.backward()
            d_optimizer.step()

            # Zero Gradients
            g_optimizer.zero_grad()

            reward = (
                torch.tensor(
                    [
                        wasserstein_distance(
                            real_outputs[i].cpu().detach().numpy(),
                            adv_outputs[i].cpu().detach().numpy(),
                        )
                        for i in range(batch_size)
                    ]
                ).to(device)
                * 100
            )

            g_loss = -torch.mean(reward * log_probs)

            # Update Generator based on Reward
            g_loss.backward()
            g_optimizer.step()

            total_d_loss += d_loss.item()
            total_g_loss += reward.mean().item()

            _, predicted = torch.max(adv_outputs, 1)
            correct_labels += (predicted == labels).sum().item()
            total_labels += labels.size(0)

            x_coords = actions[:, 0].long()
            y_coords = actions[:, 1].long()
            brightness = actions[:, 2].long() * 255
            hist, _ = np.histogram(brightness.cpu().detach().numpy(), bins=256)
            brightness_values += hist
            for x, y in zip(x_coords, y_coords):
                action_map[y, x] += 1

        train_accuracy = 100 * correct_labels / total_labels

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = discriminator(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        test_accuracy = 100 * correct / total

        action_map_np = action_map.cpu().detach().numpy()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(np.arange(256), brightness_values)

        wandb.log(
            {
                "Discriminator Loss": total_d_loss / len(train_loader),
                "Generator Loss": total_g_loss / len(train_loader),
                "Training Accuracy": train_accuracy,
                "Test Accuracy": test_accuracy,
                "Action Heatmap": [
                    wandb.Image(action_map_np, caption="Action Heatmap")
                ],
                "Brightness Histogram": [
                    wandb.Image(fig, caption="Brightness Histogram")
                ],
            }
        )

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early Stopping triggered.")
            break


# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="GAN_RL")

# Run sweep agent
wandb.agent(sweep_id, train, count=50)  # Adjust count as needed
