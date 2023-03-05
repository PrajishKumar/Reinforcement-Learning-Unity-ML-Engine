import torch
import torch.nn as nn

from utils import init_relu_weights


class Critic(nn.Module):
    """
    The neural network representing the action-value function.
    """

    def __init__(self, observation_size, action_size, num_agents, seed=0):
        """
        Set the neural network architecture.

        :param observation_size: Dimension of the observation space.
        :param action_size: Dimension of the action space.
        :param num_agents: Number of independent agents.
        :param seed: For internal random number generations.
        """
        super(Critic, self).__init__()

        self.seed = torch.manual_seed(seed)

        # Define NN architecture.
        input_size = (observation_size + action_size) * num_agents
        self.stacked_layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Initialize weights.
        self.stacked_layers.apply(init_relu_weights)

        # The last layer has different weights since it not passed through ReLU activation.
        self.stacked_layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, stacked_observations, stacked_actions):
        """
        Returns the action-values.
        :param stacked_observations: The observations of all agents stacked.
        :param stacked_actions: The actions of all agents stacked.
        :return: The action-value estimate for the corresponding observation-action pair.
        """
        return self.stacked_layers(torch.hstack((stacked_observations, stacked_actions)))
