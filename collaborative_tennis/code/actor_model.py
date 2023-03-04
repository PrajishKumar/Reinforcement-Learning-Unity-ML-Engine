import torch
import torch.nn as nn

from utils import init_relu_weights


class Actor(nn.Module):
    """
    The neural network representing the policy.
    """

    def __init__(self, observation_size, action_size, seed=0):
        """
        Set the neural network architecture.

        :param observation_size: Dimension of the observation space.
        :param action_size: Dimension of the action space.
        :param seed: For internal random number generations.
        """
        super(Actor, self).__init__()

        self.seed = torch.manual_seed(seed)

        # Define NN architecture.
        self.stacked_layers = nn.Sequential(
            # nn.BatchNorm1d(observation_size),
            nn.Linear(observation_size, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            # nn.BatchNorm1d(action_size),
            nn.Tanh(),
        )

        # Initialize weights.
        self.stacked_layers.apply(init_relu_weights)

    def forward(self, observations):
        """
        Returns the action values for an observation.
        :param observations: The observations to get the actions for.
        :return: Vector of actions in range [-1, 1].
        """
        return self.stacked_layers(observations.reshape(1, -1))
