import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import wandb
from RL_utils import sample_action_and_prob
from models import PPO_Generator, Discriminator
from Environment import Environemnt

from config import config

# Hyperparameters
epochs = 50
step_size = 5

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
    batch_size=config["batch_size"],
    shuffle=True,
)
test_loader = DataLoader(
    datasets.MNIST("./data", train=False, download=True, transform=transform),
    batch_size=config["batch_size"],
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
    generator = PPO_Generator()
    discriminator = Discriminator().to(device)

    wandb.watch((generator.model, discriminator), log="all")

    # Training Loop
    for epoch in range(epochs):
        total_d_loss = 0
        total_g_loss = 0
        correct_labels = 0
        total_labels = 0
        value_loss_s = 0

        action_map = torch.zeros((28, 28), device=device)
        env = Environemnt()

        for i, (states, labels) in enumerate(train_loader):
            generator.reset()
            for _ in range(step_size):
                states, labels = states.to(device), labels.to(device)
                action_mean, action_std, values = generator(states)
                actions, old_probs = sample_action_and_prob(action_mean, action_std)

                next_states, rewards, adv_outputs = env.step(
                    states, actions, discriminator
                )

                d_loss = discriminator.train(adv_outputs, labels)

                total_d_loss += d_loss

                generator.append(states, actions, old_probs, rewards, values)

                states = next_states

            g_loss, value_loss = generator.train()
            total_g_loss += g_loss
            value_loss_s += value_loss
            # Log Adversarial Accuracy
            _, predicted = torch.max(adv_outputs, 1)
            correct_labels += (predicted == labels).sum().item()
            total_labels += labels.size(0)

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

        x_coords = actions[:, 0].long()
        y_coords = actions[:, 1].long()

        for x, y in zip(x_coords, y_coords):
            action_map[y, x] += 1
        test_accuracy = 100 * correct / total
        print(
            "Epoch: {}, D Loss: {}, G Loss: {}, Train Accuracy: {}, Test Accuracy: {}".format(
                epoch,
                total_d_loss / len(train_loader),
                total_g_loss / len(train_loader),
                train_accuracy,
                test_accuracy,
            )
        )
        action_map_np = action_map.cpu().detach().numpy()

        wandb.log(
            {
                "Discriminator Loss": total_d_loss / len(train_loader),
                "Generator Loss": total_g_loss / len(train_loader),
                "Training Accuracy": train_accuracy,
                "Test Accuracy": test_accuracy,
                "Value Loss": value_loss_s / len(train_loader),
                "Action Heatmap": [
                    wandb.Image(action_map_np, caption="Action Heatmap")
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


def main():
    # Initialize a new sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project="GAN_RL")

    # Run sweep agent
    wandb.agent(sweep_id, train, count=50)  # Adjust count as needed
    train()


if __name__ == "__main__":
    main()
