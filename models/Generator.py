import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import copy

from config import config
from torch.nn import functional as F


# Shared Actor-Critic Model
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU(),
        )

        # Actor head (Generator)
        self.action_mean = nn.Linear(512, 3)
        self.action_std = nn.Linear(512, 3)

        # Critic head
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = self.shared(x)

        action_mean = self.action_mean(x)
        action_std = self.action_std(x)
        action_std = nn.Softplus()(action_std)  # Ensure standard deviation is positive

        value = self.value(x)

        return action_mean, action_std, value

    def get_action_prob(self, states, actions):
        action_mean, action_std, _ = self.forward(states)
        normal_distribution = torch.distributions.Normal(action_mean, action_std)
        log_prob = normal_distribution.log_prob(actions)
        return torch.exp(log_prob.sum(dim=-1))


class PPO_Generator(nn.Module):
    def __init__(
        self,
        model=ActorCritic().to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ),
        gamma=0.9,
        epsilon=0.2,
        clip_epsilon=0.1,
        c1=0.5,
        c2=0.1,
    ):
        super(PPO_Generator, self).__init__()
        self.model = model
        self.old_model = copy.deepcopy(model)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=wandb.config.g_learning_rate
        )
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.states, self.actions, self.old_probs, self.rewards, self.values = (
            [],
            [],
            [],
            [],
            [],
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, states):
        return self.model(states)

    def reset(self):
        self.states, self.actions, self.old_probs, self.rewards, self.values = (
            [],
            [],
            [],
            [],
            [],
        )

    def append(self, states, actions, old_probs, rewards, values):
        self.states.append(states)
        self.actions.append(actions)
        self.old_probs.append(old_probs)
        self.rewards.append(rewards)
        self.values.append(values)

    def train(self):
        states = torch.stack(self.states).to(self.device).float()  # [5,6000,1,28,28]
        actions = torch.stack(self.actions).to(self.device).float()  # [5,6000,3]
        old_probs = torch.stack(self.old_probs).to(self.device).float()  # [5,6000]
        rewards = torch.stack(self.rewards).to(self.device).float()  # [5,6000]
        values = torch.stack(self.values).to(self.device).squeeze(2)  # [5,6000]

        self.old_model.load_state_dict(self.model.state_dict())

        # Compute discounted rewards
        discounted_return = torch.zeros(config["batch_size"]).to(self.device)
        returns = []
        for reward in reversed(rewards):
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        returns = torch.stack(returns).to(self.device)

        advantages = returns - values
        new_probs = []
        new_values = []
        for i in range(5):
            new_prob = self.model.get_action_prob(states[i], actions[i])
            new_probs.append(new_prob)
            _, _, new_value = self.model(states[i])
            new_values.append(new_value)
        new_probs = torch.stack(new_probs).to(self.device)
        new_values = torch.stack(new_values).to(self.device).squeeze(2)
        value_loss = F.mse_loss(new_values, returns).mean()

        ratios = new_probs / old_probs

        # Compute surrogate loss
        surr1 = ratios * advantages
        surr2 = (
            torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            * advantages
        )

        policy_loss = -torch.min(surr1, surr2).mean()
        loss = policy_loss + self.c1 * value_loss
        loss = loss.float()

        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return returns.mean().item(), value_loss.item()
