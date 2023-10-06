from RL_utils import add_brightness_to_batch_images
import torch
from scipy.stats import wasserstein_distance
from config import config


class Environemnt:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pass

    def step(self, states, actions, classifier):
        _, next_states = add_brightness_to_batch_images(states, actions)
        with torch.no_grad():
            original_outputs = classifier(states)
        adv_outputs = classifier(next_states)
        reward = torch.tensor(
            [
                wasserstein_distance(
                    original_outputs[i].cpu().detach().numpy(),
                    adv_outputs[i].cpu().detach().numpy(),
                )
                for i in range(config["batch_size"])
            ]
        ).to(self.device)
        return next_states, reward, adv_outputs
